#include "../tools/server/server-page-adapters.h"

#include <cstdlib>
#include <iostream>
#include <limits>

static void require(bool condition, const char * message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

using page   = std::vector<server_page_adapter>;
using loaded = std::vector<server_loaded_adapter>;

static std::optional<std::vector<float>> decide(const page & p, const loaded & l) {
    std::string why;
    auto scales = server_page_adapter_scales(p, l, why);
    require(scales.has_value() == why.empty(), "a refusal always names its reason, a match never does");
    return scales;
}

int main() {
    const server_loaded_adapter gene  { "/genome/delia/adapter.gguf",  4365792, 100 };
    const server_loaded_adapter dream { "/engine-train/gene-b758.gguf", 4365792, 100 };

    // what this catches: the IntelMac 11-03 refusal. The page carried the gene and the dream
    // stage's runtime adapter, both at 0.0; the relaunched engine loads only the gene.
    auto s = decide({ { gene.path, 0.0f, gene.size, gene.mtime }, { dream.path, 0.0f, dream.size, dream.mtime } }, { gene });
    require(s && s->size() == 1 && (*s)[0] == 0.0f, "a page whose adapters were all at 0.0 restores under any loaded set");

    // the same lists in both: the active adapter keeps the page's scale
    s = decide({ { gene.path, 0.5f, gene.size, gene.mtime } }, { gene });
    require(s && (*s)[0] == 0.5f, "an active adapter runs at the scale the KV was computed under");

    // an adapter loaded now but absent from the page runs at 0.0, never at its default scale
    s = decide({ { gene.path, 0.5f, gene.size, gene.mtime } }, { dream, gene });
    require(s && (*s)[0] == 0.0f && (*s)[1] == 0.5f, "an extra loaded adapter restores at 0.0, in load order");

    // a page computed with no adapters restores on an engine that has some loaded
    s = decide({}, { gene });
    require(s && (*s)[0] == 0.0f, "an adapter-free page is base-model KV");

    // an active adapter that is not loaded: its KV is not this engine's
    require(!decide({ { dream.path, 1.0f, dream.size, dream.mtime } }, { gene }), "an active adapter missing from the engine refuses");

    // the same path with new weights (a gene adopted in place)
    require(!decide({ { gene.path, 1.0f, gene.size, gene.mtime - 1 } }, { gene }), "an active adapter whose file changed refuses");
    // an unreadable file proves nothing about its weights
    require(!decide({ { gene.path, 1.0f, gene.size, gene.mtime } }, { { gene.path, 0, 0 } }), "an active adapter whose file cannot be read refuses");
    // ... even when the page was saved while it was unreadable too: two unknowns never match
    require(!decide({ { gene.path, 1.0f, 0, 0 } }, { { gene.path, 0, 0 } }), "an unreadable revision never equals another");
    // an inactive adapter's revision is irrelevant: it never touched the KV
    require(decide({ { gene.path, 0.0f, 1, 1 } }, { gene }).has_value(), "an inactive adapter's revision never refuses");

    // one loaded copy cannot stand for two active entries
    require(!decide({ { gene.path, 0.5f, gene.size, gene.mtime }, { gene.path, 0.5f, gene.size, gene.mtime } }, { gene }),
            "each active entry claims its own loaded adapter");

    require(!decide({ { gene.path, std::numeric_limits<float>::quiet_NaN(), gene.size, gene.mtime } }, { gene }),
            "a non-finite recorded scale refuses");
    return 0;
}
