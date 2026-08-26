#include "da_capi.h"
#include <cstdlib>
#include <cstdio>
#include <cstring>
int main(){
    if (da_capi_abi_version() != 11) return 1;
    const char* gguf = std::getenv("DA_TEST_GGUF");
    if (!gguf) return 77;
    da_ctx* c = da_capi_load(gguf, 1);
    if (!c) { std::fprintf(stderr, "load failed\n"); return 1; }
    char* j = da_capi_info_json(c);
    bool ok = j && std::strstr(j, "embed_dim");
    std::fprintf(stderr, "info json: %s -> %s\n", j ? j : "(null)", ok ? "OK" : "FAIL");
    da_capi_free_string(j);
    // Export wrappers (best-effort): exercise glb + colmap on the native fixture.
    const char* png = std::getenv("DA_TEST_NATIVE_PNG");
    if (ok && png){
        const char* glb = "/tmp/da_capi_export.glb";
        const char* col = "/tmp/da_capi_export_colmap";
        std::remove(glb);
        int rg = da_capi_export_glb(c, png, glb);
        int rc = da_capi_export_colmap(c, png, col, 1);
        FILE* f = std::fopen(glb, "rb");
        long sz = 0; if (f){ std::fseek(f, 0, SEEK_END); sz = std::ftell(f); std::fclose(f); }
        std::fprintf(stderr, "export glb=%d (%ld bytes) colmap=%d\n", rg, sz, rc);
        if (rg != 0 || rc != 0 || sz <= 0){ ok = false; }
    }
    da_capi_free(c);
    return ok ? 0 : 1;
}
