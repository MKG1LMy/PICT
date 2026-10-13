# Sourced by the Linux launchers; do not change the caller's working directory.
pict_find_conda() {
    PICT_CONDA="${CONDA_EXE:-}"
    if [[ -n "$PICT_CONDA" && -x "$PICT_CONDA" ]]; then return 0; fi
    PICT_CONDA="$(type -P conda || true)"
    if [[ -n "$PICT_CONDA" ]]; then return 0; fi
    local candidate
    for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" /opt/conda/bin/conda /opt/miniconda3/bin/conda /opt/anaconda3/bin/conda; do
        if [[ -x "$candidate" ]]; then PICT_CONDA="$candidate"; return 0; fi
    done
    echo "Conda was not found. Set CONDA_EXE to its executable or add it to PATH." >&2
    return 1
}
