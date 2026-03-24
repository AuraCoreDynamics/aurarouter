"""Help topic registry and all built-in help content for AuraRouter.

Every topic is a ``HelpTopic`` dataclass instance registered in the
module-level ``HELP`` singleton (a ``HelpRegistry``).  The registry
supports lookup by id, keyword search, and category filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


# ------------------------------------------------------------------
# Data model
# ------------------------------------------------------------------

@dataclass(frozen=True)
class HelpTopic:
    """A single help article displayed in the Help panel."""

    id: str
    title: str
    category: str  # "concept", "panel", "howto", or "glossary"
    body: str  # HTML subset understood by QTextBrowser
    related: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------

class HelpRegistry:
    """Indexed collection of ``HelpTopic`` objects."""

    def __init__(self) -> None:
        self._topics: dict[str, HelpTopic] = {}

    # -- mutation --------------------------------------------------

    def register(self, topic: HelpTopic) -> None:
        """Add *topic* to the registry (overwrites if id already exists)."""
        self._topics[topic.id] = topic

    # -- queries ---------------------------------------------------

    def get(self, topic_id: str) -> HelpTopic | None:
        """Return the topic with *topic_id*, or ``None``."""
        return self._topics.get(topic_id)

    def search(self, query: str) -> list[HelpTopic]:
        """Return topics whose title or keywords contain *query* (case-insensitive)."""
        q = query.strip().lower()
        if not q:
            return list(self._topics.values())
        results: list[HelpTopic] = []
        for t in self._topics.values():
            if q in t.title.lower():
                results.append(t)
                continue
            if any(q in kw.lower() for kw in t.keywords):
                results.append(t)
        return results

    def by_category(self, category: str) -> list[HelpTopic]:
        """Return all topics in *category*."""
        return [t for t in self._topics.values() if t.category == category]

    def all_topics(self) -> list[HelpTopic]:
        """Return every registered topic."""
        return list(self._topics.values())

    def __len__(self) -> int:
        return len(self._topics)

    def __iter__(self) -> Iterator[HelpTopic]:
        return iter(self._topics.values())


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

HELP = HelpRegistry()


# ==================================================================
# Concept topics
# ==================================================================

HELP.register(HelpTopic(
    id="concept.moe",
    title="Mixture of Experts Routing",
    category="concept",
    keywords=["moe", "mixture", "experts", "routing", "specialist", "triage"],
    related=["concept.roles", "concept.fallback", "concept.triage"],
    body="""\
<h2>Mixture of Experts Routing</h2>
<p>AuraRouter uses a <b>Mixture of Experts (MoE)</b> pattern to direct each
task to the most appropriate model.</p>

<p><b>Analogy:</b> Think of a hospital triage desk. A nurse (the
<i>router</i>) quickly evaluates every patient and sends them to the
right specialist &mdash; a surgeon, a radiologist, or a GP. No single
doctor sees every patient, and the triage step is fast and cheap.</p>

<p>In AuraRouter the &ldquo;nurse&rdquo; is a small, fast model that
classifies your task.  The &ldquo;specialists&rdquo; are larger or
more capable models assigned to <b>roles</b> like <i>reasoning</i>,
<i>coding</i>, or <i>summarization</i>.</p>

<p>Benefits:</p>
<ul>
  <li>Small tasks stay on a cheap local model.</li>
  <li>Complex tasks automatically escalate to a more powerful model.</li>
  <li>You control cost, privacy, and latency by editing the routing
      configuration &mdash; no code changes required.</li>
</ul>
""",
))

HELP.register(HelpTopic(
    id="concept.roles",
    title="Semantic Roles",
    category="concept",
    keywords=["role", "semantic", "verb", "router", "reasoning", "coding",
              "summarization", "analysis", "reviewer"],
    related=["concept.moe", "concept.fallback", "concept.triage"],
    body="""\
<h2>Semantic Roles</h2>
<p>A <b>role</b> is a named responsibility that a model fulfills.
AuraRouter maps every incoming task to one or more roles, then sends
it to the model chain assigned to that role.</p>

<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Role</th><th>Purpose</th><th>Required?</th></tr>
<tr><td><b>router</b></td>
    <td>Intent classification and task triage</td><td>Yes</td></tr>
<tr><td><b>reasoning</b></td>
    <td>Multi-step planning and architectural reasoning</td><td>Yes</td></tr>
<tr><td><b>coding</b></td>
    <td>Code generation and implementation</td><td>Yes</td></tr>
<tr><td><b>summarization</b></td>
    <td>Text summarization and digest generation</td><td>No</td></tr>
<tr><td><b>analysis</b></td>
    <td>Data analysis and evaluation</td><td>No</td></tr>
<tr><td><b>reviewer</b></td>
    <td>Output quality assessment and correction guidance</td><td>No</td></tr>
</table>

<p>Each role also recognises <b>synonyms</b>.  For example, the words
&ldquo;planner&rdquo;, &ldquo;architect&rdquo;, and
&ldquo;planning&rdquo; all resolve to the <i>reasoning</i> role.
You can add custom roles and synonyms in <code>auraconfig.yaml</code>
under <code>semantic_verbs</code>.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.fallback",
    title="Fallback Chains",
    category="concept",
    keywords=["fallback", "chain", "priority", "retry", "resilience"],
    related=["concept.roles", "concept.tiers", "panel.routing"],
    body="""\
<h2>Fallback Chains</h2>
<p>Every role has an ordered list of models called a <b>fallback chain</b>.
When a task is routed to a role, AuraRouter tries the first model in the
chain.  If that model fails (timeout, error, overloaded), it
<i>falls back</i> to the next model, and so on.</p>

<p><b>Example:</b></p>
<pre>
  coding:
    - local_qwen        # try first (fast, free, private)
    - cloud_claude      # fallback  (powerful, costs money)
</pre>

<p>Only <b>one</b> model handles the request &mdash; the first one that
succeeds.  This gives you a predictable cost model: local-first,
cloud-only-if-needed.</p>

<p>You can reorder, add, or remove models from any chain in the
<b>Configuration &rarr; Routing</b> section of the GUI, or by editing
<code>auraconfig.yaml</code> directly.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.triage",
    title="Complexity Scoring and Triage",
    category="concept",
    keywords=["triage", "complexity", "score", "classify", "intent", "simple",
              "complex"],
    related=["concept.moe", "concept.pipeline", "concept.roles"],
    body="""\
<h2>Complexity Scoring and Triage</h2>
<p>When you submit a task, the <b>router</b> model classifies it and
assigns a <b>complexity score</b> from 1 to 10:</p>

<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Score</th><th>Meaning</th><th>Pipeline</th></tr>
<tr><td>1&ndash;3</td><td>Trivial / single-step</td>
    <td>Direct execution (skip planning)</td></tr>
<tr><td>4&ndash;7</td><td>Moderate</td>
    <td>Plan then execute</td></tr>
<tr><td>8&ndash;10</td><td>Highly complex</td>
    <td>Plan, execute, review, correct</td></tr>
</table>

<p>The classification prompt asks the router for a JSON object like
<code>{"intent": "SIMPLE_CODE", "complexity": 3}</code>.  The intent
value determines whether the task goes straight to the coding model or
through the full planning pipeline.</p>

<p>If classification fails, AuraRouter defaults to
<code>SIMPLE_CODE</code> with complexity 5 so you always get a
result.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.pipeline",
    title="Intent \u2192 Plan \u2192 Execute \u2192 Review",
    category="concept",
    keywords=["pipeline", "intent", "plan", "execute", "review", "loop",
              "correction", "dag"],
    related=["concept.triage", "concept.roles", "panel.workspace"],
    body="""\
<h2>Intent &rarr; Plan &rarr; Execute &rarr; Review</h2>
<p>Every task flows through up to four stages:</p>

<ol>
  <li><b>Intent (Classify)</b> &mdash; The <i>router</i> role reads your
      prompt and decides: is this a simple direct task or does it need
      multi-step planning?  It also produces a complexity score.</li>
  <li><b>Plan</b> &mdash; For complex tasks, the <i>reasoning</i> role
      generates an ordered list of atomic steps (returned as JSON).</li>
  <li><b>Execute</b> &mdash; The <i>coding</i> role executes each step
      in sequence.  Each step sees the output of previous steps for
      context.</li>
  <li><b>Review</b> &mdash; If a <i>reviewer</i> chain is configured,
      the output is checked for correctness.  A &ldquo;FAIL&rdquo;
      verdict triggers a <b>correction loop</b>: a new correction plan
      is generated and executed, then reviewed again &mdash; up to a
      configurable maximum number of iterations.</li>
</ol>

<p>The live DAG visualizer on the Execute tab shows each stage as a
node so you can watch progress in real time.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.privacy",
    title="Privacy Auditor and PII Detection",
    category="concept",
    keywords=["privacy", "pii", "audit", "redact", "cloud", "local",
              "sensitive", "personal"],
    related=["concept.tiers", "panel.monitor", "howto.privacy_rules"],
    body="""\
<h2>Privacy Auditor and PII Detection</h2>
<p>AuraRouter includes a <b>privacy auditor</b> that scans prompts
before they leave your machine.  It looks for patterns that may
contain personally identifiable information (PII) such as:</p>
<ul>
  <li>Email addresses</li>
  <li>Phone numbers</li>
  <li>Social Security / national ID numbers</li>
  <li>Credit card numbers</li>
  <li>Custom patterns you define</li>
</ul>

<p>When PII is detected and the task is about to be sent to a
<b>cloud</b> model, the auditor can:</p>
<ul>
  <li><b>Warn</b> you in the Privacy tab.</li>
  <li><b>Skip</b> cloud models and force the task to stay on-prem.</li>
  <li><b>Redact</b> matches before sending (configurable).</li>
</ul>

<p>All detections are logged in the <b>Privacy</b> tab so you have an
audit trail.  You can add custom regex patterns under
<code>privacy.patterns</code> in <code>auraconfig.yaml</code>.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.tiers",
    title="Hosting Tiers",
    category="concept",
    keywords=["tier", "hosting", "on-prem", "cloud", "dedicated", "local",
              "tenant"],
    related=["concept.fallback", "concept.privacy", "concept.providers"],
    body="""\
<h2>Hosting Tiers</h2>
<p>Every model in your configuration has a <b>hosting tier</b> that
tells AuraRouter where it runs:</p>

<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Tier</th><th>Description</th><th>Example</th></tr>
<tr><td><b>on-prem</b></td>
    <td>Runs on your machine or local network. Data never leaves
    your control.</td>
    <td>Ollama, llama.cpp</td></tr>
<tr><td><b>dedicated-tenant</b></td>
    <td>Cloud infrastructure reserved for your organization.
    Shared nothing with other tenants.</td>
    <td>Private Azure / GCP deployments</td></tr>
<tr><td><b>cloud</b></td>
    <td>Shared multi-tenant cloud API. Fastest to set up, but data
    leaves your network.</td>
    <td>Claude API, Gemini API</td></tr>
</table>

<p>The privacy auditor uses tiers to decide whether to allow, warn,
or block a request.  Fallback chains typically list on-prem models
first so that data stays local whenever possible.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.providers",
    title="Built-in and External Providers",
    category="concept",
    keywords=["provider", "ollama", "llama", "llamacpp", "openapi", "cloud",
              "package", "plugin"],
    related=["concept.catalog", "concept.tiers", "howto.first_model",
             "howto.external_provider"],
    body="""\
<h2>Built-in and External Providers</h2>
<p>A <b>provider</b> is the backend that actually runs inference.
AuraRouter ships with three built-in providers:</p>

<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Provider</th><th>Config Key</th><th>Notes</th></tr>
<tr><td>Ollama</td><td><code>ollama</code></td>
    <td>Connects to a running Ollama server via HTTP.</td></tr>
<tr><td>llama.cpp Server</td><td><code>llamacpp-server</code></td>
    <td>Connects to a llama.cpp HTTP server you start separately.</td></tr>
<tr><td>OpenAPI-Compatible</td><td><code>openapi</code></td>
    <td>Works with any endpoint that implements the OpenAI chat
    completions API.</td></tr>
</table>

<p>For proprietary cloud models, install an <b>external provider
package</b>:</p>
<ul>
  <li><code>pip install aurarouter-claude</code> &mdash; Anthropic Claude</li>
  <li><code>pip install aurarouter-gemini</code> &mdash; Google Gemini</li>
</ul>

<p>External packages register themselves automatically via Python
entry points.  Once installed, their models appear in the provider
catalog.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.catalog",
    title="Provider Catalog",
    category="concept",
    keywords=["catalog", "entry point", "discovery", "register", "plugin",
              "package"],
    related=["concept.providers", "howto.external_provider", "panel.models"],
    body="""\
<h2>Provider Catalog</h2>
<p>The <b>provider catalog</b> is how AuraRouter discovers available
inference backends at startup.</p>

<h3>Automatic discovery</h3>
<p>External provider packages (like <code>aurarouter-claude</code>)
register a Python <b>entry point</b> in the
<code>aurarouter.providers</code> group.  AuraRouter scans these
entry points on launch and adds each provider to the catalog
automatically.</p>

<h3>Manual registration</h3>
<p>You can also register a provider manually in
<code>auraconfig.yaml</code> by specifying its module path under
<code>providers.custom</code>.  This is useful for internal or
experimental backends.</p>

<h3>Model discovery</h3>
<p>Once a provider is registered, AuraRouter can query it for
available models.  For Ollama, this means calling
<code>/api/tags</code>; for cloud providers, each package defines
its own discovery method.  Discovered models appear in the
<b>Models</b> tab.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.mcp",
    title="MCP Tools Overview",
    category="concept",
    keywords=["mcp", "model context protocol", "tool", "server", "client"],
    related=["panel.settings", "concept.pipeline"],
    body="""\
<h2>MCP Tools Overview</h2>
<p><b>MCP</b> (Model Context Protocol) is a standard for connecting AI
models to tool-providing servers.  AuraRouter can run as an MCP
server, exposing its routing capabilities as <b>tools</b> that any
MCP-compatible client can call.</p>

<p>Built-in MCP tools:</p>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Tool</th><th>Description</th></tr>
<tr><td><b>Route Task</b></td>
    <td>General-purpose router &mdash; sends a prompt through the
    full Intent &rarr; Plan &rarr; Execute pipeline.</td></tr>
<tr><td><b>Local Inference</b></td>
    <td>Privacy-preserving execution on local models only. Cloud
    models are never tried.</td></tr>
<tr><td><b>Generate Code</b></td>
    <td>Multi-step code generation with planning.</td></tr>
<tr><td><b>Compare Models</b></td>
    <td>Run the same prompt across multiple models and return all
    results for comparison.</td></tr>
</table>

<p>You can enable or disable individual tools in the
<b>Configuration</b> tab under <i>MCP Tools</i>.  Changes take
effect after saving and restarting the MCP server.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.grid",
    title="AuraGrid Integration",
    category="concept",
    keywords=["auragrid", "grid", "distributed", "mas", "cell", "node",
              "deploy"],
    related=["concept.tiers", "howto.grid_deploy", "panel.settings"],
    body="""\
<h2>AuraGrid Integration</h2>
<p><b>AuraGrid</b> is a federated compute fabric that orchestrates
services across multiple machines.  When the optional
<code>aurarouter[auragrid]</code> package is installed, AuraRouter
can run as a <b>Managed Application Service (MAS)</b> on an AuraGrid
cell.</p>

<p>What this means in practice:</p>
<ul>
  <li><b>Environment selector</b> in the toolbar lets you switch
      between Local and AuraGrid at runtime.</li>
  <li><b>Deployment panel</b> lets you pick a deployment strategy
      (single-node, replicated, or partitioned).</li>
  <li><b>Cell status panel</b> shows live health for every node in
      your cell.</li>
  <li><b>Configuration propagation</b> &mdash; when you save config
      in AuraGrid mode, changes are pushed to all nodes in the
      cell.</li>
</ul>

<p>AuraGrid integration is entirely optional.  AuraRouter works
perfectly well as a standalone local application or MCP server.</p>
""",
))


# ==================================================================
# Panel topics
# ==================================================================

HELP.register(HelpTopic(
    id="panel.workspace",
    title="Execute Tab (Workspace)",
    category="panel",
    keywords=["execute", "workspace", "task", "prompt", "dag", "history",
              "output", "input"],
    related=["concept.pipeline", "concept.triage", "panel.routing"],
    body="""\
<h2>Execute Tab (Workspace)</h2>
<p>The Execute tab is your primary workspace for running tasks.</p>

<h3>Task Input</h3>
<p>Type a natural-language description of what you need. You can also
attach files or paste context into the <i>Context</i> field and select
an <i>Output Format</i> (text, markdown, python, etc.).</p>

<h3>Running a Task</h3>
<p>Click <b>Execute</b> (or press <b>Ctrl+Enter</b>). The router
classifies your task, optionally generates a plan, then executes it.
Progress appears in the status bar and the DAG visualizer.</p>

<h3>DAG Visualizer</h3>
<p>The collapsible graph below the input shows each pipeline stage
(Classify, Plan, Step 1, Step 2, Review&hellip;) as a node. Green
nodes succeeded; red nodes failed.  Hover for details.</p>

<h3>Prompt History</h3>
<p>The <i>Recent Tasks</i> dropdown stores your last 20 tasks with
their results.  Select one to restore both the prompt and the
output.</p>

<h3>Keyboard Shortcuts</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Shortcut</th><th>Action</th></tr>
<tr><td>Ctrl+Enter</td><td>Execute task</td></tr>
<tr><td>Ctrl+N</td><td>Clear all fields for a new prompt</td></tr>
<tr><td>Escape</td><td>Cancel running execution</td></tr>
</table>
""",
))

HELP.register(HelpTopic(
    id="panel.routing",
    title="Routing Editor (Configuration Tab)",
    category="panel",
    keywords=["routing", "chain", "editor", "role", "fallback", "config",
              "configuration"],
    related=["concept.fallback", "concept.roles", "panel.settings"],
    body="""\
<h2>Routing Editor</h2>
<p>The <b>Routing</b> section of the Configuration tab lets you
manage fallback chains for each role.</p>

<h3>Reading the Table</h3>
<p>Each row shows a role name and its fallback chain. The chain reads
left to right: the first model is tried first, the second is the
fallback, and so on. Models are separated by <code>&gt;</code>.</p>

<h3>Editing a Chain</h3>
<ol>
  <li>Select a role from the dropdown (or type a new role name).</li>
  <li>Pick a model from the <i>Add model</i> dropdown.</li>
  <li>Click <b>Append</b> to add it to the end of the chain.</li>
  <li>Use <b>Up</b> / <b>Down</b> to reorder.</li>
  <li>Use <b>Remove from Chain</b> to drop the last model.</li>
</ol>

<h3>Required Roles</h3>
<p>The three required roles (<i>router</i>, <i>reasoning</i>,
<i>coding</i>) are highlighted if missing. You must configure at
least these three for the routing pipeline to work.</p>

<p>Click <b>Save</b> at the bottom to persist changes to
<code>auraconfig.yaml</code>.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.models",
    title="Models Tab",
    category="panel",
    keywords=["model", "catalog", "download", "gguf", "huggingface",
              "manage", "browse"],
    related=["concept.providers", "concept.catalog", "howto.first_model"],
    body="""\
<h2>Models Tab</h2>
<p>The Models tab lets you browse, download, and manage models.</p>

<h3>Local Models</h3>
<p>Lists GGUF model files stored on your machine. You can see file
size, quantization, and which roles reference each model.</p>

<h3>HuggingFace Downloads</h3>
<p>Search for GGUF models on HuggingFace Hub, pick a quantization
variant, and download directly. Requires the <code>[local]</code>
install extra.</p>

<h3>Provider Catalog</h3>
<p>If external provider packages are installed (e.g.
<code>aurarouter-claude</code>), their available models also appear
here.  Select one to add it to your configuration.</p>

<h3>AuraGrid Mode</h3>
<p>When connected to AuraGrid, this tab shows models available across
the cell rather than just local files.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.monitor",
    title="Traffic and Privacy Tabs (Monitor)",
    category="panel",
    keywords=["traffic", "privacy", "monitor", "token", "cost", "audit",
              "health", "dashboard"],
    related=["concept.privacy", "howto.privacy_rules", "howto.budget"],
    body="""\
<h2>Traffic and Privacy Tabs</h2>

<h3>Traffic Tab</h3>
<p>Shows token usage and estimated cost across all models:</p>
<ul>
  <li><b>Token counters</b> &mdash; input and output tokens per model.</li>
  <li><b>Cost estimates</b> &mdash; based on the pricing catalog in your
      config. Local models show as $0.</li>
  <li><b>Request history</b> &mdash; timestamped log of every inference
      call with latency and status.</li>
</ul>

<h3>Privacy Tab</h3>
<p>The privacy audit log records every PII detection event:</p>
<ul>
  <li>Which pattern matched (email, phone, custom, etc.).</li>
  <li>Which model the prompt was headed to.</li>
  <li>What action was taken (warned, blocked, redacted).</li>
</ul>

<p>Use this tab to verify that sensitive data is staying on-prem.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.settings",
    title="Configuration Tab (Settings)",
    category="panel",
    keywords=["settings", "config", "yaml", "mcp", "budget", "save",
              "revert", "configuration"],
    related=["panel.routing", "panel.models", "concept.mcp"],
    body="""\
<h2>Configuration Tab</h2>
<p>The Configuration tab is the central place for managing your
AuraRouter setup.</p>

<h3>Models Section</h3>
<p>Add, edit, or remove model definitions.  Each model has an ID,
provider, endpoint, optional tags, and a hosting tier.</p>

<h3>Routing Section</h3>
<p>Manage fallback chains for each role. See the
<i>Routing Editor</i> help topic for details.</p>

<h3>MCP Tools</h3>
<p>Toggle individual MCP tools on or off.  Disabled tools are not
exposed to MCP clients. Changes take effect after saving and
restarting the server.</p>

<h3>YAML Preview</h3>
<p>The right-hand pane shows a live preview of the YAML that will be
written to <code>auraconfig.yaml</code> when you save.  You can
copy it to the clipboard.</p>

<h3>Save and Revert</h3>
<p><b>Save</b> writes changes to disk and reloads the routing fabric.
<b>Revert</b> discards unsaved changes and reloads from disk.
In AuraGrid mode, saving propagates config to all cell nodes.</p>
""",
))


# ==================================================================
# How-to topics
# ==================================================================

HELP.register(HelpTopic(
    id="howto.first_model",
    title="How to Set Up Your First Model",
    category="howto",
    keywords=["first", "setup", "ollama", "quickstart", "install", "start",
              "beginner"],
    related=["concept.providers", "panel.models", "panel.routing"],
    body="""\
<h2>Set Up Your First Model (Ollama)</h2>
<p>The fastest way to get started is with <b>Ollama</b>, a free
local model runner.</p>

<h3>Step 1 &mdash; Install Ollama</h3>
<p>Download from <code>https://ollama.com</code> and follow the
installer.  After installation, open a terminal and run:</p>
<pre>ollama pull qwen2.5-coder:7b</pre>
<p>This downloads a capable 7-billion-parameter coding model.</p>

<h3>Step 2 &mdash; Configure AuraRouter</h3>
<p>Open the <b>Configuration</b> tab. In the <i>Models</i> section,
click <b>Add</b> and fill in:</p>
<ul>
  <li><b>Model ID:</b> <code>local_qwen</code></li>
  <li><b>Provider:</b> <code>ollama</code></li>
  <li><b>Endpoint:</b> <code>http://localhost:11434/api/generate</code></li>
  <li><b>Model Name:</b> <code>qwen2.5-coder:7b</code></li>
  <li><b>Hosting Tier:</b> <code>on-prem</code></li>
</ul>

<h3>Step 3 &mdash; Assign Roles</h3>
<p>In the <i>Routing</i> section, add <code>local_qwen</code> to the
<b>router</b>, <b>reasoning</b>, and <b>coding</b> chains.  A single
model can serve all three roles to start.</p>

<h3>Step 4 &mdash; Save and Test</h3>
<p>Click <b>Save</b>, switch to the <b>Execute</b> tab, type a short
prompt like &ldquo;Write a Python hello-world script&rdquo;, and press
<b>Ctrl+Enter</b>.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.cloud_fallback",
    title="How to Add a Cloud Fallback",
    category="howto",
    keywords=["cloud", "fallback", "claude", "gemini", "api", "key"],
    related=["concept.fallback", "concept.tiers", "howto.external_provider"],
    body="""\
<h2>Add a Cloud Fallback</h2>
<p>A cloud fallback ensures complex tasks succeed even when your local
model struggles.</p>

<h3>Step 1 &mdash; Install a Provider Package</h3>
<pre>pip install aurarouter-claude</pre>
<p>(or <code>aurarouter-gemini</code> for Google Gemini)</p>

<h3>Step 2 &mdash; Add Your API Key</h3>
<p>Set the environment variable for your provider:</p>
<pre>
# Claude
export ANTHROPIC_API_KEY="sk-..."

# Gemini
export GOOGLE_API_KEY="..."
</pre>

<h3>Step 3 &mdash; Register the Model</h3>
<p>In the Configuration tab, click <b>Add</b> in the Models section.
The provider package should appear in the provider dropdown.  Enter
the model name (e.g. <code>claude-sonnet-4-20250514</code>) and set the
hosting tier to <code>cloud</code>.</p>

<h3>Step 4 &mdash; Append to Chains</h3>
<p>In the Routing section, select each role where you want cloud
fallback, pick the new cloud model, and click <b>Append</b>.  Make
sure the cloud model is <i>after</i> your local model in the
chain.</p>

<h3>Step 5 &mdash; Save</h3>
<p>Click <b>Save</b>.  Now if your local model fails, the cloud model
takes over automatically.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.external_provider",
    title="How to Install an External Provider",
    category="howto",
    keywords=["external", "provider", "install", "package", "plugin",
              "catalog", "entry point"],
    related=["concept.providers", "concept.catalog", "howto.cloud_fallback"],
    body="""\
<h2>Install an External Provider Package</h2>
<p>External providers add support for cloud APIs that are not included
in the base AuraRouter install.</p>

<h3>Step 1 &mdash; Install the Package</h3>
<pre>pip install aurarouter-claude</pre>
<p>Available packages:</p>
<ul>
  <li><code>aurarouter-claude</code> &mdash; Anthropic Claude</li>
  <li><code>aurarouter-gemini</code> &mdash; Google Gemini</li>
</ul>

<h3>Step 2 &mdash; Verify Discovery</h3>
<p>Restart AuraRouter (or the GUI).  The new provider should appear
in the <b>Models</b> tab under the provider catalog.  If it does not,
check that the package installed into the same Python environment that
AuraRouter is running in.</p>

<h3>Step 3 &mdash; Set Credentials</h3>
<p>Each provider package documents its required environment variables
(e.g. <code>ANTHROPIC_API_KEY</code>).  Set them before launching
AuraRouter.</p>

<h3>Step 4 &mdash; Add a Model</h3>
<p>In the Configuration tab, click <b>Add</b>.  Select the new
provider, enter the model name, and set the hosting tier.  Then
add it to your fallback chains as needed.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.privacy_rules",
    title="How to Create Custom Privacy Rules",
    category="howto",
    keywords=["privacy", "regex", "pattern", "pii", "custom", "rule",
              "redact"],
    related=["concept.privacy", "panel.monitor"],
    body="""\
<h2>Create Custom Privacy Rules</h2>
<p>AuraRouter ships with built-in PII patterns, but you can add your
own.</p>

<h3>Step 1 &mdash; Open auraconfig.yaml</h3>
<p>Find (or create) the <code>privacy</code> section:</p>
<pre>
privacy:
  enabled: true
  action: warn          # warn | block | redact
  patterns:
    - name: internal_id
      regex: "PROJ-\\d{5,8}"
      description: "Internal project identifiers"
    - name: badge_number
      regex: "BADGE-[A-Z0-9]{6}"
      description: "Employee badge numbers"
</pre>

<h3>Step 2 &mdash; Define Your Pattern</h3>
<p>Each pattern needs:</p>
<ul>
  <li><b>name</b> &mdash; unique label for the audit log.</li>
  <li><b>regex</b> &mdash; Python-compatible regular expression.</li>
  <li><b>description</b> &mdash; human-readable note.</li>
</ul>

<h3>Step 3 &mdash; Choose an Action</h3>
<p>The <code>action</code> field controls what happens when any
pattern matches:</p>
<ul>
  <li><b>warn</b> &mdash; log the event but send the prompt anyway.</li>
  <li><b>block</b> &mdash; skip cloud models entirely.</li>
  <li><b>redact</b> &mdash; replace matched text with
      <code>[REDACTED]</code> before sending.</li>
</ul>

<h3>Step 4 &mdash; Save and Verify</h3>
<p>Save the config (GUI or file), run a task containing your pattern,
and check the <b>Privacy</b> tab to confirm detection.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.budget",
    title="How to Set Spending Limits",
    category="howto",
    keywords=["budget", "spending", "limit", "cost", "cap", "money",
              "pricing"],
    related=["panel.monitor", "concept.tiers"],
    body="""\
<h2>Set Spending Limits</h2>
<p>You can cap how much AuraRouter spends on cloud API calls.</p>

<h3>Step 1 &mdash; Configure Pricing</h3>
<p>In <code>auraconfig.yaml</code>, add pricing info for your cloud
models:</p>
<pre>
models:
  cloud_claude:
    provider: claude
    model_name: claude-sonnet-4-20250514
    hosting_tier: cloud
    pricing:
      input_per_1k: 0.003
      output_per_1k: 0.015
</pre>

<h3>Step 2 &mdash; Set a Budget</h3>
<pre>
budget:
  daily_limit_usd: 5.00
  monthly_limit_usd: 50.00
  action: block          # block | warn
</pre>

<p>When the limit is reached:</p>
<ul>
  <li><b>block</b> &mdash; cloud models are skipped; only local models
      are used.</li>
  <li><b>warn</b> &mdash; a warning is logged but requests continue.</li>
</ul>

<h3>Step 3 &mdash; Monitor</h3>
<p>The <b>Traffic</b> tab shows cumulative spending.  Local models
always show $0 since they run on your own hardware.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.grid_deploy",
    title="How to Deploy on AuraGrid",
    category="howto",
    keywords=["auragrid", "deploy", "grid", "cell", "mas", "distributed"],
    related=["concept.grid", "concept.tiers"],
    body="""\
<h2>Deploy AuraRouter on AuraGrid</h2>
<p>Running AuraRouter as a Managed Application Service on AuraGrid
gives you distributed access to your routing fabric.</p>

<h3>Prerequisites</h3>
<ul>
  <li>A running AuraGrid cell (see AuraGrid documentation).</li>
  <li><code>pip install aurarouter[auragrid]</code></li>
</ul>

<h3>Step 1 &mdash; Switch Environment</h3>
<p>In the GUI toolbar, open the <i>Environment</i> dropdown and select
<b>AuraGrid</b>.  If a local service is running, you will be asked to
stop it first.</p>

<h3>Step 2 &mdash; Choose a Deployment Strategy</h3>
<p>The Deployment panel (visible in AuraGrid mode) offers:</p>
<ul>
  <li><b>Single-node</b> &mdash; one instance on one node.</li>
  <li><b>Replicated</b> &mdash; identical copies across nodes for
      high availability.</li>
  <li><b>Partitioned</b> &mdash; different roles run on different
      nodes.</li>
</ul>

<h3>Step 3 &mdash; Configure and Save</h3>
<p>Set your models and routing chains as usual.  When you click
<b>Save</b>, the config is propagated to every node in the cell.</p>

<h3>Step 4 &mdash; Start the Service</h3>
<p>Click <b>Start</b> in the toolbar.  AuraGrid will schedule the
MAS across your cell according to the chosen strategy.</p>
""",
))
