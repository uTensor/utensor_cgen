## The Architecture of `utensor_cgen`

![utensor-cli-components](images/utensor-cli-components.drawio.svg)

- `utensor_cgen` consists of multiple **components**, such as frontend component and backend component
- Each component consists of multiple **parts**. For example, a backend may be composed of a graph lower part and a code generator part
- Most of the compoents found in the graph above is customizable. The user can register their own component to `utensor_cgen`

## Examples

- `backend_registration.ipynb`: example notebook for backend registration, this allows you to inject customized backend to the `utensor-cli`
- `frontend_registration.ipynb`: example notebook for frontend parser registration, this allows you to inject customized frontend parser to the `utensor-cli`
- `transformer_registration.ipynb`: example notebook for transformer registration, this allows you to inject customized transformer into the transformation pipeline in `utensor-cli`

## Load Plugin

- with `utensor-cli`
    - You can use `--plugin` flag to load the plugins
    - It's a multiple flag. So you can pass multiple values to the cli
        - ex: `utensor-cli --plugin plugin1 --plugin plugin2 [convert|generate-config|...]`
        - both `plugin1` and `plugin2` will be loaded
        - Note that `--plugin` must be passed before any subcommand such as `convert`
- with script
    - simply import your plugins and it should work

```python
import plugin1, plugin2
import utensor_cgen

# do stuffs with utensor_cgen and your plugins
```