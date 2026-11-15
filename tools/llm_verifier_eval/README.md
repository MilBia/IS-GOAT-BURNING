# llm_verifier_eval

Offline harness for asking: can a (small, local) vision LLM verify a fire alarm raised by the classic colour+motion gate,
the way Gemini does in `hybrid_cloud`? It sends labelled frames to any OpenAI-compatible `/chat/completions` endpoint
(LiteLLM, llama.cpp server, LM Studio, ...) and reports alarm rates per category, latency, parse failures,
a threshold sweep and repeat-stability. Images are **not** kept in the repository.

```bash
export LITELLM_BASE_URL=http://HOST:PORT/v1   # never commit this or the key
export LITELLM_API_KEY=...                    # optional for servers without auth

# 1. Frames from videos (label: yes = fire, no = no fire; category names are free text; categories starting
#    with "synth" are excluded from the headline recall, categories starting with "goat" form the "goat" FPR).
python -m tools.llm_verifier_eval frames --src clips/fire  --label yes --cat real_fire     --out frames/fire  --manifest set.json
python -m tools.llm_verifier_eval frames --src clips/goat  --label no  --cat goat_neg      --out frames/goat  --manifest set.json
python -m tools.llm_verifier_eval frames --src clips/hard  --label no  --cat hard_neg_real --out frames/hard  --manifest set.json

# 2. Query (resumable, appends JSON lines). The image is sent first, then the prompt.
python -m tools.llm_verifier_eval run --manifest set.json --model chat --out res.jsonl --width 640 \
    --prompts config strict yesno
python -m tools.llm_verifier_eval run --manifest stab.json --model chat --out res.jsonl --width 640 --repeats 3 --stability-n 8

# 3. Markdown report
python -m tools.llm_verifier_eval report res.jsonl
```

Prompts: `config` = the default `GEMINI__PROMPT` plus a JSON-output instruction; `strict` = scene-aware prompt that lists the
usual false-positive sources; `minimal`; `yesno` = one-word answer, scored by `P(yes)/(P(yes)+P(no))` from first-token logprobs
(needs a server that returns `logprobs`), which allows a threshold sweep. The self-reported JSON `confidence` of small models is
nearly constant and should not be thresholded.

Notes: temperature is 0, so repeated runs of one server are expected to be (near) deterministic; "stability" mostly checks that.
Image-token cost depends on the width (`--width`); on a CPU-only llama.cpp server the image encode dominates latency.

Caveats: a response that cannot be parsed counts as "no alarm" in the tables (its number is in each header as `unparsable`);
transport errors are excluded from the denominators and are retried when `run` is resumed; resuming is keyed by
(image, prompt, repeat, width, model), so several models can share one results file; frames of one clip are correlated, so the
Wilson intervals assume independence they do not have and are too narrow; the API key is sent in clear text over `http://`,
so use a trusted network or `https://`. The tool reads `LITELLM_*` directly from the environment (it is outside the app and
does not use `settings`). `frames` also takes `--per-clip` and `--start`, `run` takes `--timeout`.
