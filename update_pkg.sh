pip uninstall -y diff-surfel-rasterization
rm -rf submodules/diff-surfel-rasterization/build
rm -rf submodules/diff-surfel-rasterization/diff_surfel_renderer.egg-info
pip install -e submodules/diff-surfel-rasterization --no-cache-dir