-- Host wrappers for the preserved gfx936 AOT device code. No kernel regeneration.
-- The staged reviewer build supplies H3_AOT_DEPENDENCIES; a plain checkout uses
-- its own public sources. Neither layout needs an author's precompiled .a file.
local dependencies = os.getenv("H3_AOT_DEPENDENCIES")
    or path.join(os.projectdir(), "tools/h3_submission/support/dependencies")
local dtk = os.getenv("DTK_ROOT") or "/opt/dtk"

for _, kind in ipairs({"qknorm", "qkrope"}) do
    local generated = path.join(dependencies, kind, "generated")
    target("h3-" .. kind .. "-aot")
        set_default(false)
        set_kind("static")
        set_languages("c11", "cxx17")
        add_cxflags("-fPIC", "-O3", {force = true})
        add_defines("__HIP_PLATFORM_AMD__")
        add_includedirs(path.join(dtk, "include"), generated, {public = true})
        add_files(path.join(generated, "*.cpp"), path.join(generated, kind .. ".c"))
        set_targetdir(path.join(dependencies, kind, "build"))
        on_load(function (target)
            local expected = kind == "qknorm" and 12 or 8
            local sources = os.files(path.join(generated, "*.cpp"))
            assert(#sources == expected, "H3 " .. kind .. ": expected " .. expected .. " public C++ sources under " .. generated)
            assert(os.isfile(path.join(generated, kind .. ".c")), "H3 dispatch source missing: " .. generated)
            assert(os.isfile(path.join(dependencies, "abi/ninetoothed.h")), "H3 NineToothed ABI header missing: " .. dependencies)
        end)
    target_end()
end
