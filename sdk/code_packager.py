"""
CodePackager - Handles packaging of Python code and dependencies
"""

import zipfile
import tempfile
import os
import inspect
import ast
import importlib.util
import json
from typing import List, Dict, Any, Callable, Set
import sys


class CodePackager:
    """
    Packages Python functions and their dependencies into a ZIP file for remote execution.
    """
    
    def __init__(self):
        self.imported_modules: Set[str] = set()
        self.source_files: Dict[str, str] = {}
    
    def package_function(self, 
                        func: Callable, 
                        requirements: List[str],
                        job_id: str,
                        args: List[Any] = None,
                        **kwargs) -> bytes:
        """
        Package a function and its dependencies into a ZIP file.
        
        Args:
            func: Function to package
            requirements: List of required packages
            job_id: Unique job identifier
            args: List of positional arguments to pass to the function
            **kwargs: Additional keyword arguments to pass to the function
            
        Returns:
            ZIP file as bytes
        """
        # Create temporary ZIP file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add runtime helper
                runtime_content = self._get_runtime_helper()
                zip_file.writestr('runtime.py', runtime_content)
                
                # Add main execution script
                main_script = self._generate_main_script(func, job_id, args or [], **kwargs)
                zip_file.writestr('main.py', main_script)
                
                # Add requirements
                requirements_content = '\n'.join(requirements) if requirements else ''
                zip_file.writestr('requirements.txt', requirements_content)
                
                # Add function source code
                func_source = self._extract_function_source(func)
                zip_file.writestr('function.py', func_source)
                
                # Add any additional source files
                for filename, content in self.source_files.items():
                    zip_file.writestr(filename, content)
                
                # Add job configuration
                job_config = {
                    'job_id': job_id,
                    'function_name': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                zip_file.writestr('job_config.json', json.dumps(job_config, indent=2))
            
            # Read ZIP file as bytes
            with open(temp_file.name, 'rb') as f:
                zip_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return zip_data
    
    def _generate_main_script(self, func: Callable, job_id: str, args: List[Any], **kwargs) -> str:
        """Generate the main execution script."""
        return f'''#!/usr/bin/env python3
"""
Auto-generated execution script for job {job_id}
"""

import sys
import json
import os
import traceback
from function import {func.__name__}

def main():
    """Main execution function."""
    try:
        # Load job configuration
        with open('job_config.json', 'r') as f:
            config = json.load(f)
        
        print(f"🚀 Starting job {{config['job_id']}}")
        print(f"Function: {{config['function_name']}}")
        
        # Import and call the function
        args = config.get('args', [])
        kwargs = config.get('kwargs', {{}})
        result = {func.__name__}(*args, **kwargs)
        
        # Save result
        with open('result.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print("✅ Job completed successfully")
        
    except Exception as e:
        error_info = {{
            'error': str(e),
            'traceback': traceback.format_exc()
        }}
        
        with open('error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"❌ Job failed: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
    
    def _extract_function_source(self, func: Callable) -> str:
        """Extract the source code of a function and its dependencies."""
        try:
            # Get the source code of the function
            source = inspect.getsource(func)
            
            # Parse the AST to find imports and other dependencies
            tree = ast.parse(source)
            
            # Extract imports
            imports = self._extract_imports(tree)
            
            # Build the complete source file
            complete_source = '\n'.join(imports) + '\n\n' + source
            
            return complete_source
            
        except Exception as e:
            # Fallback: just return the function source with proper indentation
            source = inspect.getsource(func)
            # Remove any leading indentation issues
            lines = source.split('\n')
            if lines and lines[0].startswith('    '):
                # Remove common indentation
                min_indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
                lines = [line[min_indent:] if line.strip() else line for line in lines]
            return '\n'.join(lines)
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements from AST."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                if node.names:
                    names = ', '.join([alias.name for alias in node.names])
                    imports.append(f"from {module} import {names}")
        
        return list(set(imports))  # Remove duplicates
    
    def add_source_file(self, filename: str, content: str):
        """Add an additional source file to the package."""
        self.source_files[filename] = content
    
    def _get_runtime_helper(self) -> str:
        """Get the runtime helper code."""
        return '''"""
Cumulus Runtime Helper - Provides checkpointing and pause/resume functionality
"""

import os
import json
import time
import torch
from typing import Dict, Any, Optional, List


def job_dir():
    """Get the job directory from environment variable."""
    return os.getenv('CUMULUS_JOB_DIR', os.getcwd())


def _control_path():
    """Get the path to the control file."""
    return os.path.join(job_dir(), 'control.json')


def should_pause() -> bool:
    """Check if the job should pause."""
    p = _control_path()
    if not os.path.exists(p): 
        return False
    try:
        with open(p, 'r') as f:
            return bool(json.load(f).get('pause', False))
    except Exception:
        return False


class Checkpointer:
    """Handles model checkpointing and resuming."""
    
    def __init__(self, fname: str = 'checkpoint.pt'):
        self.path = os.path.join(job_dir(), fname)
        self._last_ts = 0

    def exists(self):
        """Check if checkpoint exists."""
        return os.path.exists(self.path)

    def save(self, model, optimizer, epoch: int, step: int, extra: dict = None):
        """Save model checkpoint."""
        state = {
            'epoch': epoch,
            'step': step,
            'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'optimizer': optimizer.state_dict(),
            'rng_cpu': torch.random.get_rng_state(),
            'rng_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'extra': extra or {}
        }
        torch.save(state, self.path)
        self._last_ts = time.time()
        return self.path

    def load(self, model, optimizer):
        """Load model checkpoint."""
        state = torch.load(self.path, map_location='cpu')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        torch.random.set_rng_state(state['rng_cpu'])
        if torch.cuda.is_available() and state.get('rng_cuda') is not None:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        return state

    def time_to_checkpoint(self, step: int, every_steps: int = None, every_seconds: int = None):
        """Check if it's time to checkpoint."""
        by_step = (every_steps is not None and step > 0 and step % every_steps == 0)
        by_time = (every_seconds is not None and (time.time() - self._last_ts) >= every_seconds)
        return by_step or by_time


def list_checkpoints() -> List[Dict[str, Any]]:
    """List available checkpoints in the job directory."""
    checkpoints = []
    job_d = job_dir()
    
    for fname in os.listdir(job_d):
        if fname.endswith('.pt'):
            fpath = os.path.join(job_d, fname)
            try:
                state = torch.load(fpath, map_location='cpu')
                checkpoints.append({
                    'filename': fname,
                    'path': fpath,
                    'epoch': state.get('epoch', 0),
                    'step': state.get('step', 0),
                    'timestamp': os.path.getmtime(fpath)
                })
            except Exception:
                continue
    
    return sorted(checkpoints, key=lambda x: x['timestamp'], reverse=True)
'''
    
    def package_directory(self, directory_path: str, requirements: List[str]) -> bytes:
        """
        Package an entire directory for remote execution.
        
        Args:
            directory_path: Path to directory to package
            requirements: List of required packages
            
        Returns:
            ZIP file as bytes
        """
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_file:
            with zipfile.ZipFile(temp_file.name, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add all Python files from directory
                for root, dirs, files in os.walk(directory_path):
                    for file in files:
                        if file.endswith('.py'):
                            file_path = os.path.join(root, file)
                            arcname = os.path.relpath(file_path, directory_path)
                            zip_file.write(file_path, arcname)
                
                # Add requirements
                requirements_content = '\n'.join(requirements) if requirements else ''
                zip_file.writestr('requirements.txt', requirements_content)
                
                # Add execution script
                exec_script = self._generate_directory_exec_script()
                zip_file.writestr('main.py', exec_script)
            
            # Read ZIP file as bytes
            with open(temp_file.name, 'rb') as f:
                zip_data = f.read()
            
            # Clean up
            os.unlink(temp_file.name)
            
            return zip_data
    
    def _generate_directory_exec_script(self) -> str:
        """Generate execution script for directory packages."""
        return '''#!/usr/bin/env python3
"""
Auto-generated execution script for directory package
"""

import sys
import json
import os
import traceback

def main():
    """Main execution function."""
    try:
        print("🚀 Starting directory package execution")
        
        # Look for main.py or __main__.py
        if os.path.exists('main.py'):
            exec(open('main.py').read())
        elif os.path.exists('__main__.py'):
            exec(open('__main__.py').read())
        else:
            print("❌ No main.py or __main__.py found")
            sys.exit(1)
        
        print("✅ Package execution completed successfully")
        
    except Exception as e:
        error_info = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        
        with open('error.json', 'w') as f:
            json.dump(error_info, f, indent=2)
        
        print(f"❌ Package execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
