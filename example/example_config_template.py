from config_manager import config_field, config_template


class ExampleConfigTemplate:

    base_template = config_template.Template(
        fields=[config_field.Field(name="num_iterations", types=[int])],
        nested_templates=[],
    )
