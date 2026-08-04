# Fail if a test source exists on disk but is never named in the tests manifest.
#
# What this catches: a tests/test-*.cpp that compiles nowhere and runs nowhere,
# yet reads as coverage. Hit for real -- test-moe-residency.cpp sat in tests/ at
# 54 KB, fully written, never registered in CMakeLists.txt, so not one of its
# assertions had ever executed.
#
# References inside comments do NOT count as registration: a commented-out
# llama_build_and_test() line is precisely the state this guard exists to catch.
# A file that is deliberately not built must be named in the allow-list, with a
# comment saying why.
#
# Inputs:
#   CHECK_TESTS_DIR                 directory holding the test-*.cpp sources
#   CHECK_TESTS_MANIFEST            the CMakeLists.txt that must name them
#   CHECK_TESTS_ALLOW_UNREGISTERED  file names deliberately not built
#
# Runs standalone (no compiler, no configured build tree):
#   cmake -DCHECK_TESTS_DIR=tests -DCHECK_TESTS_MANIFEST=tests/CMakeLists.txt \
#         -P cmake/check-tests-registered.cmake

foreach (_var CHECK_TESTS_DIR CHECK_TESTS_MANIFEST)
    if (NOT DEFINED ${_var})
        message(FATAL_ERROR "check-tests-registered.cmake: ${_var} is required")
    endif()
endforeach()

if (NOT EXISTS "${CHECK_TESTS_MANIFEST}")
    message(FATAL_ERROR "check-tests-registered.cmake: no such manifest: ${CHECK_TESTS_MANIFEST}")
endif()

# Strip whole-line comments so a commented-out registration cannot satisfy the check.
file(STRINGS "${CHECK_TESTS_MANIFEST}" _manifest_lines)
set(_manifest_active "")
foreach (_line IN LISTS _manifest_lines)
    if (NOT _line MATCHES "^[ \t]*#")
        string(APPEND _manifest_active "${_line}\n")
    endif()
endforeach()

file(GLOB _test_sources RELATIVE "${CHECK_TESTS_DIR}" "${CHECK_TESTS_DIR}/test-*.cpp")

set(_dead "")
foreach (_src IN LISTS _test_sources)
    if (_src IN_LIST CHECK_TESTS_ALLOW_UNREGISTERED)
        continue()
    endif()
    string(FIND "${_manifest_active}" "${_src}" _pos)
    if (_pos EQUAL -1)
        list(APPEND _dead "${_src}")
    endif()
endforeach()

if (_dead)
    string(REPLACE ";" "\n    " _dead_pretty "${_dead}")
    message(FATAL_ERROR
        "Unregistered test source(s) -- these would never compile and never run:\n"
        "    ${_dead_pretty}\n"
        "Register each with llama_build_and_test(<file>) in ${CHECK_TESTS_MANIFEST}, "
        "or add it to LLAMA_TESTS_UNREGISTERED there with a comment saying why it is not built.")
endif()
