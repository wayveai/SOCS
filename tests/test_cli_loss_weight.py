import runpy
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize('arguments, expected', [([], 1e-4), (['--bc_loss_weight', '0.25'], 0.25)])
def test_training_cli_forwards_numeric_behavioral_loss_weight(monkeypatch, arguments, expected):
    monkeypatch.setattr('sys.argv', ['main.py', '--behavioral_cloning_task', *arguments])
    with patch('model.SOCS') as model, patch('pytorch_lightning.Trainer'), \
            patch('pytorch_lightning.callbacks.ModelCheckpoint'), \
            patch('pytorch_lightning.loggers.TensorBoardLogger'):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')

    assert model.call_args.kwargs.get('bc_loss_weight') == expected


def test_training_cli_rejects_non_numeric_behavioral_loss_weight(monkeypatch):
    monkeypatch.setattr('sys.argv', ['main.py', '--bc_loss_weight', 'invalid'])
    with patch('model.SOCS'), patch('pytorch_lightning.Trainer'), \
            patch('pytorch_lightning.callbacks.ModelCheckpoint'), \
            patch('pytorch_lightning.loggers.TensorBoardLogger'), pytest.raises(SystemExit):
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'main.py'), run_name='__main__')
