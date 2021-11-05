import example_config_template
from config_manager import base_configuration


class ExampleConfig(base_configuration.BaseConfiguration):
    def __init__(self, config, changes):
        super().__init__(
            configuration=config,
            changes=changes,
            template=example_config_template.ExampleConfigTemplate.base_template,
        )
