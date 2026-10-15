# shellcheck shell=bash disable=SC2034
# Paths shared by the evaluation scripts (sourced). Set ORL_EVAL_WORK; everything else defaults to a directory in it.
ORL_EVAL=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
: "${ORL_EVAL_WORK:?set ORL_EVAL_WORK to a directory for images, runtimes, inputs, models, generations and scores}"
ORL_EVAL_WORK=$(realpath -m "$ORL_EVAL_WORK")
IMAGES=$(realpath -m "${ORL_EVAL_IMAGES:-$ORL_EVAL_WORK/images}")
case "$ORL_EVAL$ORL_EVAL_WORK$IMAGES" in
  *[,:]*) echo "evaluation paths must not contain ',' or ':' (Apptainer bind syntax)" >&2; exit 2 ;;
esac
RUNTIMES=$ORL_EVAL_WORK/runtime
INPUTS=$ORL_EVAL_WORK/inputs
MODELS=$ORL_EVAL_WORK/models
RECIPES=$ORL_EVAL_WORK/recipes
RUNS=$ORL_EVAL_WORK/runs
SCORING=$ORL_EVAL_WORK/scoring
SCORER_IMAGE=python@sha256:a3ab0b966bc4e91546a033e22093cb840908979487a9fc0e6e38295747e49ac0
# host paths visible inside the containers; add more (comma-separated) with ORL_EVAL_BIND
BINDS="$ORL_EVAL,$ORL_EVAL_WORK${ORL_EVAL_BIND:+,$ORL_EVAL_BIND}"
export PYTHONDONTWRITEBYTECODE=1
