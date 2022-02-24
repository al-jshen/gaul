import jax.numpy as jnp
from jx.nn import transformer


class TestTransformerShapes:
    def __init__(self) -> None:
        self.num_heads = 8
        self.model_size = 256
        self.feedforward_size = 512
        self.batch_size = 16
        self.num_layers = 2
        self.input_vocab_size = 100
        self.target_vocab_size = 101
        self.input_position_encoding = 150
        self.target_position_encoding = 151
        self.dropout_rate = 0.2
        self.input_sequence_length = 17
        self.target_sequence_length = 26
        self.transformer = transformer.Transformer(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            model_size=self.model_size,
            feedforward_size=self.feedforward_size,
            dropout_rate=self.dropout_rate,
            input_vocab_size=self.input_vocab_size,
            target_vocab_size=self.target_vocab_size,
            input_position_encoding=self.input_position_encoding,
            target_position_encoding=self.target_position_encoding,
        )

    def test_encoder_decoder_layers(self):
        enc = self.transformer.encoder.encoding_layers[0]
        x1 = jnp.ones((self.batch_size, self.input_sequence_length, self.model_size))
        y = enc(x1)
        assert y.shape == (self.batch_size, self.input_sequence_length, self.model_size)

        dec = self.transformer.decoder.decoding_layers[0]
        x2 = jnp.ones((self.batch_size, self.target_sequence_length, self.model_size))
        z = dec(x2, y)
        assert z.shape == (
            self.batch_size,
            self.target_sequence_length,
            self.model_size,
        )

    def test_encoder_decoder(self):
        x1 = jnp.ones((self.batch_size, self.input_sequence_length))
        y = self.transformer.encoder(x1)
        assert y.shape == (
            self.batch_size,
            self.target_sequence_length,
            self.model_size,
        )

        x2 = jnp.ones((self.batch_size, self.target_sequence_length))
        z = self.transformer.decoder(x2, y)
        assert z.shape == (
            self.batch_size,
            self.target_sequence_length,
            self.model_size,
        )

    def test_transformer(self):
        x = jnp.ones((self.batch_size, self.input_sequence_length))
        y = jnp.ones((self.batch_size, self.target_sequence_length))
        z = self.transformer(x, y)
        assert z.shape == (
            self.batch_size,
            self.target_sequence_length,
            self.model_size,
        )
