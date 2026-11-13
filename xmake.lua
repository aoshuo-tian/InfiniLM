option("ninetoothed-hygon-qkrope")
    set_default(false)
    set_showmenu(true)
    set_description("Experimental H3 D128 Q/K norm plus partial M-RoPE")
option_end()
if has_config("ninetoothed-hygon-qkrope") then
    add_defines("INFINILM_ENABLE_NINETOOTHED_HYGON_QKROPE")
end

option("ninetoothed-hygon-qknorm")
    set_default(false)
    set_showmenu(true)
    set_description("Experimental D128 Hygon NineToothed Q/K normalization")
option_end()
if has_config("ninetoothed-hygon-qknorm") then
    add_defines("INFINILM_ENABLE_NINETOOTHED_HYGON_QKNORM")
end

if os.getenv("H3_REVIEW_VENDOR_HEADERS") then
    add_includedirs(path.join(os.projectdir(), "../../tooling/pybind11/include"), {system = true})
else
    add_requires("pybind11")
end

set_toolchains("gcc")

option("cxx11-abi")
    set_default(nil)
    set_showmenu(true)
    set_description("Set _GLIBCXX_USE_CXX11_ABI to match the installed InfiniCore")
    set_values("0", "1")
option_end()

local cxx11_abi = os.getenv("INFINILM_CXX11_ABI") or get_config("cxx11-abi")
if cxx11_abi and cxx11_abi ~= "" then
    if cxx11_abi ~= "0" and cxx11_abi ~= "1" then
        raise("INFINILM_CXX11_ABI must be 0 or 1")
    end
    add_defines("_GLIBCXX_USE_CXX11_ABI=" .. cxx11_abi)
end

-- Add spdlog from third_party directory
add_includedirs("third_party/spdlog/include")
add_includedirs("third_party/json/single_include/")

option("ninetoothed-hygon-adaln")
    set_default(false)
    set_showmenu(true)
    set_description("Route H3 AdaLN through the optional matching InfiniCore operator")
option_end()
if has_config("ninetoothed-hygon-adaln") then
    add_defines("INFINILM_ENABLE_NINETOOTHED_HYGON_ADALN")
end

option("ninetoothed-hygon-gated-residual")
    set_default(false)
    set_showmenu(true)
    set_description("Enable guarded NineToothed Hygon gated residual fusion")
option_end()

if has_config("ninetoothed-hygon-gated-residual") then
    add_defines("INFINILM_ENABLE_NINETOOTHED_HYGON_GATED_RESIDUAL")
end

includes("xmake/h3_support.lua")

target("_infinilm")
    if os.getenv("H3_REVIEW_VENDOR_HEADERS") then
        add_includedirs(os.getenv("H3_REVIEW_PYTHON_INCLUDE"), {system = true})
        add_linkdirs(os.getenv("H3_REVIEW_PYTHON_LIBDIR"))
        add_links(os.getenv("H3_REVIEW_PYTHON_LINK"))
    else
        add_packages("pybind11")
    end
    set_default(false)
    add_rules("python.module", {soabi = true})
    set_languages("cxx17")
    set_kind("shared")

    local INFINI_ROOT = os.getenv("INFINI_ROOT") or (os.getenv(is_host("windows") and "HOMEPATH" or "HOME") .. "/.infini")

    -- add_includedirs("csrc", { public = false })
    -- add_includedirs("csrc/pybind11", { public = false })
    add_includedirs(INFINI_ROOT.."/include", { public = true })
    -- spdlog is already included globally via add_includedirs at the top

    add_linkdirs(INFINI_ROOT.."/lib")
    add_links("infinicore_cpp_api", "infiniop", "infinirt", "infiniccl")

    -- Add C++ sources
    add_files("csrc/**.cpp")
    add_files("csrc/**.cc")

    set_installdir("python/infinilm")
    -- Both wrappers are compiled even when the experimental QKRoPE route is off.
    -- Their AOT sources are public inputs, not pre-existing archives in an AFS tree.
    add_deps("h3-qknorm-aot", "h3-qkrope-aot")
    add_linkdirs(path.join(os.getenv("DTK_ROOT") or "/opt/dtk", "lib"))
    add_syslinks("amdhip64")
    add_ldflags("-Wl,-z,defs", {force = true})
target_end()


option("ninetoothed-hygon-projection-swiglu")
    set_default(false)
    set_showmenu(true)
    set_description("Use matching InfiniCore H3 projection/SwiGLU fusion")
option_end()
if has_config("ninetoothed-hygon-projection-swiglu") then
    add_defines("INFINILM_ENABLE_NINETOOTHED_HYGON_PROJECTION_SWIGLU")
end
