# MobileNet V2 para classificação de metástase em linfonodos axilares (ALN)
"""
Módulo MobileNet para classificação de patches histológicos de câncer de mama.

Este módulo implementa um pipeline completo usando MobileNetV2 pré-treinado
para classificar patches em três categorias de metástase ALN:
- N0: Sem metástase
- N+(1-2): 1-2 linfonodos com metástase
- N+(>2): Mais de 2 linfonodos com metástase

Principais componentes:
- mobilenet_pipeline.py: Classe principal MobileNetALNClassifier
- mobilenet_train.py: Script de treinamento com interface de usuário
"""

from .mobilenet_pipeline import MobileNetALNClassifier

__all__ = ['MobileNetALNClassifier']
__version__ = '1.0.0'
