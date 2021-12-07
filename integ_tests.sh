pytest tests -m init -v
#pytest tests -v -m component --config_path "runs/migration_python_gen/config.yaml"
for config_path in  $(find runs -type f -name *.yaml)
        do
          pytest tests -v -m component --config_path "$config_path"
        done