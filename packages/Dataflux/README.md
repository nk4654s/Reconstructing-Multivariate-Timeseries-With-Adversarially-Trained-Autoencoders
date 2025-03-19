# Dataflux
## Usage
The package can be executed as standalone using `flux-pipeline`. In general, the `flux-pipeline` downloads a dataset 
from the Amazon S3 bucket, runs some preprocessing and stores the files on the local disk. Afterward, the data is 
further processed (e.g. normalize the dataset). The recommended way to execute the pipeline is by configuring a 
`yaml`-config file.

```yaml
folder: <output folder name>
machine_uuid: <machine UUID>
intervals:
  - <start date (optional time> <end date (optional time)>
  - ...
num_workers: <processing threads>
overwrite: <overwrite existing folder>

modules:
  - class_path: Dataflux.modules.State
    init_args:
      key: <sensor UUID>
      lower: <split data where sensor is lower equal than x>
      upper: <split data where sensor is higher equal than x>
      alpha_threshold: <seconds to remove after split>
      beta_threshold: <seconds to remove before split>
  - ...
```

## Modules
### Available Modules
```yaml
- Dataflux.modules.features.Features
- Dataflux.modules.filter.Filter
- Dataflux.modules.normalize.[NormalizeFit | NormalizeTransform]
- Dataflux.modules.rename.Rename
- Dataflux.modules.slice.Slice
- Dataflux.modules.state.State
```

### Adding Modules
> Note: A detailed description of the named classes can be found in the developer documentation. The guide on how
> to generate the `docs/` folder can be found in the repository's main `README.md` or click [here](https://github.com/neaxplore/xplore-ai-cas/blob/master/README.md).

Adding a new module can be achieved, by creating a new class in `Dataflux.modules` with 
`Dataflux.diskio.FileWorkerCallable` as base class. The base class serves as interface and requires to implement the
following methods.

```python
class FileWorkerCallable:
    def get_dependencies(self) -> Dict[str, CallableDependency]:
        return {}

    def finish(self, context: FileWorkerContext, metadata: Metadata) -> None:
        raise NotImplementedError

    def run(self, fm: FileMeta, context: FileWorkerContext) -> CALLABLE_RETURN_TYPE:
        raise NotImplementedError
```
> Note: Even though `get_dependencies` is not necessary, adding dependencies to each module can help building a valid
> pipeline. To learn more about `CallableDependency`, head to the developer documentation.

An instance of a module will be created for each processing thread defined in the configuration. Therefore, the
`run()` method will be executed on a subset of dataset files downloaded. The number of files an instance processes
can be calculated by

$$\hat{n}_\textit{files}=\frac{n_\textit{files}}{m_\textit{threads}}$$

> Note: The algorithm responsible for assigning the files to each thread can be found in `Dataflux.diskio.factory`.

