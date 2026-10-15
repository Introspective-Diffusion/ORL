FROM vllm/vllm-openai@sha256:9c719fc0c869092c7d0533f8357d6985a38d5ff03b20ffb6a4620c2b4806dd4b
ARG SAMPLER=base
COPY inference/vllm/patches/ /opt/idlm-patches/
RUN case "$SAMPLER" in base|orl) ;; *) exit 2 ;; esac \
    && python3 /opt/idlm-patches/vllm_dg_hf_align_patch.py \
    && if [ "$SAMPLER" = orl ]; then python3 /opt/idlm-patches/vllm_dg_adaptive_scalar_patch.py; fi
