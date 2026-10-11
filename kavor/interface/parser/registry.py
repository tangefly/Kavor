PARSER_REGISTRY = {}


def register_parser(model_type):

    def decorator(cls):
        PARSER_REGISTRY[model_type] = cls
        return cls

    return decorator