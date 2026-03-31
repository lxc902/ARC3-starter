action:
	uv run ARC-AGI-3-Agents/main.py --agent=action

install:
	uv venv
	cd ARC-AGI-3-Agents && UV_PROJECT_ENVIRONMENT=../.venv uv sync --all-extras
	uv pip install -r requirements.txt
	$(MAKE) patch-submodule

patch-submodule:
	@python3 -c "\
	import pathlib; \
	p = pathlib.Path('ARC-AGI-3-Agents/agents/__init__.py'); \
	t = p.read_text(); \
	marker = 'from custom_agent import'; \
	patch = 'import sys, os\nsys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))\nfrom custom_agent import *\n'; \
	done = marker in t; \
	print('  ✓ Already patched') if done else (p.write_text(t.replace('load_dotenv()', patch + '\nload_dotenv()')), print('  ✓ Patch applied')); \
	"

setup-env:
	@if [ ! -f ARC-AGI-3-Agents/.env ]; then \
		cp ARC-AGI-3-Agents/.env.example ARC-AGI-3-Agents/.env; \
		echo "Created .env — edit ARC-AGI-3-Agents/.env and set your ARC_API_KEY"; \
	else \
		echo ".env already exists"; \
	fi

viewer:
	.venv/bin/python viewer.py

tensorboard:
	.venv/bin/tensorboard --logdir=runs --port=6006

clean:
	rm -rf ./runs ./recordings
