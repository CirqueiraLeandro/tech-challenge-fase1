"""Utilitários para notebooks - imports e paths seguros"""

from pathlib import Path
import sys


def setup_notebook_environment():
    """Setup do ambiente do notebook"""
    # Adicionar src ao sys.path
    project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    return project_root


def get_data_path(filename: str = "Telco_customer_churn.xlsx") -> Path:
    """Retorna caminho seguro para arquivo de dados"""
    project_root = Path.cwd().parent if 'notebooks' in str(Path.cwd()) else Path.cwd()
    data_path = project_root / "data" / "raw" / filename
    
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {data_path}\n"
            f"Diretório atual: {Path.cwd()}\n"
            f"Arquivos em data/raw/: {list((project_root / 'data' / 'raw').glob('*'))}"
        )
    
    return data_path


def get_project_root() -> Path:
    """Retorna raiz do projeto"""
    if 'notebooks' in str(Path.cwd()):
        return Path.cwd().parent
    return Path.cwd()
