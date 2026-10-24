import shutil
from pathlib import Path

import pytest
from utils import *

# Continuum fork: a loaded gene (LoRA adapter) paged between the device and host RAM, and dropped
# (server-genes.h; continuum docs/architecture/SERVING-ENGINE.md, the `genes` module).

LORA_FILE_URL = "https://huggingface.co/ggml-org/stories15M_MOE/resolve/main/moe_shakespeare15M.gguf"
TRAIN_DIR = Path("./tmp/lora-page-train")
GENE = "shakespeare.gguf"
PROMPT = "Look in thy glass"

server = ServerPreset.stories15m_moe()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.stories15m_moe()
    server.server_metrics = True
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(download_file(LORA_FILE_URL), TRAIN_DIR / GENE)
    server.train_dir = str(TRAIN_DIR)


def load_gene():
    res = server.make_request("POST", "/lora-adapters/load", data={"name": GENE})
    assert res.status_code == 200, res.body
    return res.body["id"]


def page(action, gene_id):
    return server.make_request("POST", f"/lora-adapters/{action}", data={"id": gene_id})


def complete(gene_id, scale):
    res = server.make_request("POST", "/completion", data={
        "prompt": PROMPT, "lora": [{"id": gene_id, "scale": scale}],
        "seed": 42, "temperature": 0.0, "n_predict": 24, "cache_prompt": False,
    })
    assert res.status_code == 200, res.body
    return res.body["content"]


def listed(gene_id):
    res = server.make_request("GET", "/lora-adapters")
    assert res.status_code == 200
    return next(entry for entry in res.body if entry["id"] == gene_id)


def metric(name):
    res = server.make_request("GET", "/metrics")
    assert res.status_code == 200
    for line in res.body.splitlines():
        if line.startswith(f"llamacpp:{name} "):
            return float(line.split()[1])
    raise AssertionError(f"metric {name} missing")


def test_page_out_in_is_identical_to_never_paging():
    # what this catches: a page_in that re-binds the tensors to the wrong bytes or the wrong buffer
    # type (the gene's output changes), a page_out that does not actually free its device bytes, or
    # sizes and timings that are not reported where the core's working-set owner reads them.
    global server
    server.start()
    assert server.make_request("GET", "/props").body["lora_page_version"] == 1
    gene = load_gene()
    reference = complete(gene, 1.0)
    assert reference != complete(gene, 0.0), "the gene must change the output for this test to see it"
    on_device = listed(gene)
    assert on_device["resident"] == "device" and on_device["n_bytes"] > 0

    out = page("page_out", gene)
    assert out.status_code == 200, out.body
    assert out.body["resident"] == "host" and out.body["n_moved"] == out.body["n_bytes"] > 0
    assert out.body["t_ms"] >= 0
    assert listed(gene)["resident"] == "host"
    assert metric("lora_host_bytes") == pytest.approx(out.body["n_bytes"], rel=1e-5)  # /metrics prints 6 digits
    assert metric("lora_device_bytes") == 0
    assert metric("lora_page_out_total") == 1

    back = page("page_in", gene)
    assert back.status_code == 200, back.body
    assert back.body["resident"] == "device" and back.body["n_moved"] == out.body["n_bytes"]
    assert metric("lora_host_bytes") == 0
    assert complete(gene, 1.0) == reference
    assert metric("lora_page_in_on_demand_total") == 0, "an explicit page_in is not a miss"


def test_a_task_applying_a_paged_out_gene_pages_it_in_first():
    # what this catches: a decode computing with an adapter whose device buffers were freed (a
    # crash, or garbage) when the core has not paged it back in ahead of the turn. The task pages it
    # in on its own path, produces the same output, and the miss is counted for the core to see.
    global server
    server.start()
    gene = load_gene()
    reference = complete(gene, 1.0)
    assert page("page_out", gene).status_code == 200
    assert complete(gene, 1.0) == reference
    assert listed(gene)["resident"] == "device"
    assert metric("lora_page_in_on_demand_total") == 1
    # a task that does not apply it leaves it where it is
    assert page("page_out", gene).status_code == 200
    complete(gene, 0.0)
    assert listed(gene)["resident"] == "host"
    assert metric("lora_page_in_on_demand_total") == 1


def test_page_refusals_are_typed_and_repeats_are_idempotent():
    # what this catches: paging out an adapter every default request runs (it would page back in
    # on every request and thrash), silently accepting an unknown id, and a retried page_out
    # copying twice.
    global server
    server.start()
    gene = load_gene()
    assert page("page_out", 99).status_code == 400
    assert page("page_in", 99).status_code == 400

    assert server.make_request("POST", "/lora-adapters", data=[{"id": gene, "scale": 1.0}]).status_code == 200
    refused = page("page_out", gene)
    assert refused.status_code == 400 and "active by default" in refused.body["error"]["message"]
    assert server.make_request("POST", "/lora-adapters", data=[{"id": gene, "scale": 0.0}]).status_code == 200

    first = page("page_out", gene)
    again = page("page_out", gene)
    assert first.status_code == again.status_code == 200
    assert again.body["n_moved"] == 0 and again.body["n_bytes"] == first.body["n_bytes"]
    assert metric("lora_page_out_total") == 1
    assert page("page_in", gene).body["n_moved"] > 0
    assert page("page_in", gene).body["n_moved"] == 0


def test_unload_drops_a_paged_out_gene():
    # what this catches: a gene dropped while in host RAM leaking its host copy (the host gauge
    # stays up) or staying addressable.
    global server
    server.start()
    gene = load_gene()
    assert page("page_out", gene).status_code == 200
    dropped = server.make_request("POST", "/lora-adapters/unload", data={"id": gene})
    assert dropped.status_code == 200 and dropped.body["retired"] is True
    assert all(entry["id"] != gene for entry in server.make_request("GET", "/lora-adapters").body)
    assert metric("lora_host_bytes") == 0
    assert page("page_in", gene).status_code == 400
