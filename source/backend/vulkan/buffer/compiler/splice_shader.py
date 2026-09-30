#!/usr/bin/env python3
"""Compile ONE buffer-backend shader and splice it into the checked-in tables.

makeshader.py regenerates every shader, which (a) rewrites all of the 8 MB
AllShader.cpp whenever the local glslang / spirv-opt differ from the ones
upstream used, and (b) aborts on hosts whose glslang predates extensions such
as GL_KHR_cooperative_matrix. This helper runs the same pipeline as
makeshader.py (same headers, glslangValidator -V, spirv-opt -O, xxd -i) for a
single .comp and its variants, then replaces or appends only those entries in
AllShader.cpp, ../shaders/AllShader.h and VulkanShaderMap.cpp.

Usage (from this directory):  python3 splice_shader.py conv3d.comp
Variants follow macro.json (useFP16 / macro list), like makeshader.py.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
GLSL = os.path.join(HERE, "..", "execution", "glsl")
SRC = os.path.join(HERE, "AllShader.cpp")
HDR = os.path.join(HERE, "..", "shaders", "AllShader.h")
MAP = os.path.join(HERE, "VulkanShaderMap.cpp")

sys.path.insert(0, HERE)
import makeshader  # noqa: E402  (reuse FP32_HEADER / FP16_HEADER)


def variants(comp, macros):
    cfg = macros.get(comp, {})
    macro_list = cfg.get("macro", []) or []
    out = [(None, "fp32")] + [(m, "fp32") for m in macro_list]
    if cfg.get("useFP16", False):
        out += [(None, "fp16")] + [(m, "fp16") for m in macro_list]
    return out


def symbol(comp, macro, precision):
    name = "glsl_" + comp[: -len(".comp")]
    if macro:
        name += "_" + macro
    if precision == "fp16":
        name += "_FP16"
    return name + "_comp"


def compile_one(body, macro, precision, tmp):
    src = os.path.join(tmp, "shader.comp")
    header = makeshader.FP16_HEADER if precision == "fp16" else makeshader.FP32_HEADER
    with open(src, "w") as f:
        f.write(header + "\n" + ("#define " + macro + "\n" if macro else "") + body)
    env = ["--target-env", "vulkan1.1"] if "GL_KHR_shader_subgroup" in body else []
    raw, spv = os.path.join(tmp, "raw.spv"), os.path.join(tmp, "opt.spv")
    subprocess.run(["glslangValidator", "-V"] + env + [src, "-o", raw], check=True)
    if shutil.which("spirv-opt") and os.environ.get("MNN_VULKAN_DISABLE_SPIRV_OPT", "") != "1":
        subprocess.run(["spirv-opt", "-O", raw, "-o", spv], check=True)
    else:
        spv = raw
    with open(spv, "rb") as f:
        return f.read()


def c_array(name, data):
    lines = []
    for i in range(0, len(data), 12):
        lines.append("  " + ", ".join("0x%02x" % b for b in data[i:i + 12]))
    return ("const unsigned char %s[] = {\n%s\n};\nunsigned int %s_len = %d;\n"
            % (name, ",\n".join(lines), name, len(data)))


def splice(comp):
    with open(os.path.join(GLSL, "macro.json")) as f:
        macros = json.load(f)
    with open(os.path.join(GLSL, comp)) as f:
        body = f.read()
    with open(SRC) as f:
        src = f.read()
    with open(HDR) as f:
        hdr = f.read()
    with open(MAP) as f:
        mp = f.read()
    tmp = tempfile.mkdtemp()
    try:
        for macro, precision in variants(comp, macros):
            name = symbol(comp, macro, precision)
            arr = c_array(name, compile_one(body, macro, precision, tmp))
            pat = re.compile(r"const unsigned char %s\[\] = \{.*?\};\nunsigned int %s_len = \d+;\n"
                             % (re.escape(name), re.escape(name)), re.S)
            if pat.search(src):
                src = pat.sub(lambda _m: arr, src)
            else:
                src = src.rstrip("\n") + "\n" + arr
            decl = ("extern const unsigned char %s[];\nextern unsigned int %s_len;\n" % (name, name))
            if decl not in hdr:
                hdr = hdr.replace("#endif", decl + "#endif", 1) if hdr.rstrip().endswith("#endif") else hdr + decl
            entry = ('mMaps.insert(std::make_pair("%s", std::make_pair(%s,%s_len)));\n' % (name, name, name))
            if entry not in mp:
                idx = mp.rstrip().rfind("}\n}")
                mp = mp[:idx] + entry + mp[idx:]
            print("spliced", name)
    finally:
        shutil.rmtree(tmp)
    with open(SRC, "w") as f:
        f.write(src)
    with open(HDR, "w") as f:
        f.write(hdr)
    with open(MAP, "w") as f:
        f.write(mp)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    splice(sys.argv[1])
