from talking_avatar import generator as generator_module


def test_talking_avatar_generator_wraps_runtime(monkeypatch):
    init_kwargs = {}
    generate_kwargs = {}

    class Runtime:
        def generate(self, **kwargs):
            generate_kwargs.update(kwargs)
            return "frames"

    def fake_factory(**kwargs):
        init_kwargs.update(kwargs)
        return Runtime()

    monkeypatch.setattr(generator_module, "create_generation_runtime", fake_factory)
    generator = generator_module.TalkingAvatarGenerator(component="value")
    result = generator.generate(prompt="hello")

    assert init_kwargs == {"component": "value"}
    assert generate_kwargs == {"prompt": "hello"}
    assert result == "frames"
