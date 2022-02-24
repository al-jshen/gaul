import jax.numpy as jnp
from jacket.nn.transformer import Transformer


num_heads = 8
model_size = 256
feedforward_size = 512
batch_size = 16
num_layers = 2
input_vocab_size = 100
target_vocab_size = 101
input_position_encoding = 150
target_position_encoding = 151
dropout_rate = 0.2
input_sequence_length = 17
target_sequence_length = 26

transformer = Transformer(
    num_heads=num_heads,
    model_size=model_size,
    feedforward_size=feedforward_size,
    num_layers=num_layers,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    input_position_encoding=input_position_encoding,
    target_position_encoding=target_position_encoding,
    dropout_rate=dropout_rate,
)


class TestTransformerShapes:
    def test_encoder_decoder_layers(self):
        enc = transformer.encoder.encoding_layers[0]
        x1 = jnp.ones((batch_size, input_sequence_length, model_size))
        y = enc(x1)
        assert y.shape == (batch_size, input_sequence_length, model_size)

        dec = transformer.decoder.decoding_layers[0]
        x2 = jnp.ones((batch_size, target_sequence_length, model_size))
        z = dec(x2, y)
        assert z.shape == (
            batch_size,
            target_sequence_length,
            model_size,
        )

    def test_encoder_decoder(self):
        x1 = jnp.ones((batch_size, input_sequence_length))
        y = transformer.encoder(x1)
        assert y.shape == (
            batch_size,
            input_sequence_length,
            model_size,
        )

        x2 = jnp.ones((batch_size, target_sequence_length))
        z = transformer.decoder(x2, y)
        assert z.shape == (
            batch_size,
            target_sequence_length,
            model_size,
        )

    def test_transformer(self):
        x = jnp.ones((batch_size, input_sequence_length))
        y = jnp.ones((batch_size, target_sequence_length))
        z = transformer(x, y)
        assert z.shape == (
            batch_size,
            target_sequence_length,
            target_vocab_size,
        )
