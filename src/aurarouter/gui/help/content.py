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
    title="Unified Artifact Catalog",
    category="concept",
    keywords=["catalog", "artifact", "entry point", "discovery", "register",
              "plugin", "package", "model", "service", "analyzer"],
    related=["concept.providers", "concept.analyzers", "howto.external_provider",
             "panel.models"],
    body="""\
<h2>Unified Artifact Catalog</h2>
<p>The <b>Unified Artifact Catalog</b> is a single typed registry that
manages three kinds of artifact:</p>

<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Kind</th><th>Description</th><th>Example</th></tr>
<tr><td><b>model</b></td>
    <td>An inference model &mdash; local or cloud.</td>
    <td><code>local_qwen</code>, <code>cloud_claude</code></td></tr>
<tr><td><b>service</b></td>
    <td>An external MCP service endpoint (e.g. AuraGrid service,
    remote tool server).</td>
    <td><code>grid-summarizer</code></td></tr>
<tr><td><b>analyzer</b></td>
    <td>A route analyzer that controls how tasks are classified
    and dispatched to roles.</td>
    <td><code>aurarouter-default</code>, <code>auraxlm-remote</code></td></tr>
</table>

<p>All three kinds live in the <code>catalog</code> section of
<code>auraconfig.yaml</code>.  Legacy entries in the <code>models</code>
section are surfaced transparently as <code>kind: model</code>
artifacts, so existing configs continue to work without changes.</p>

<h3>Automatic discovery</h3>
<p>External provider packages (like <code>aurarouter-claude</code>)
register a Python <b>entry point</b> in the
<code>aurarouter.providers</code> group.  AuraRouter scans these
entry points on launch and adds each provider to the catalog
automatically.</p>

<h3>Manual registration</h3>
<p>You can register artifacts manually in <code>auraconfig.yaml</code>
under the <code>catalog</code> section, or use the CLI:</p>
<pre>aurarouter catalog register my-model --kind model --display-name "My Model"</pre>

<h3>Querying the catalog</h3>
<p>The catalog supports filtered queries by kind, tags, capabilities,
and provider.  Use <code>aurarouter catalog artifacts --kind analyzer</code>
or the MCP tool <code>aurarouter.catalog.list</code>.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.analyzers",
    title="Route Analyzers",
    category="concept",
    keywords=["analyzer", "route", "triage", "dispatch", "active",
              "built-in", "remote", "auraxlm"],
    related=["concept.catalog", "concept.triage", "concept.pipeline",
             "howto.analyzer"],
    body="""\
<h2>Route Analyzers</h2>
<p>A <b>route analyzer</b> is the component that decides how each
incoming task is classified, which role handles it, and what
execution strategy to use.  It sits at the very top of the routing
pipeline.</p>

<p><b>Analogy:</b> If roles are the doctors in a hospital, the
analyzer is the <i>triage protocol</i> &mdash; it decides which
doctor handles each patient and how urgently.  Switching the analyzer
changes the triage rules without touching the doctors or the
treatments.</p>

<h3>Built-in analyzer</h3>
<p>AuraRouter ships with a built-in analyzer called
<code>aurarouter-default</code>.  It uses intent classification and
complexity scoring to map tasks to roles:</p>
<ul>
  <li><code>simple_code</code> &rarr; <i>coding</i> role</li>
  <li><code>complex_reasoning</code> &rarr; <i>reasoning</i> role</li>
  <li><code>review</code> &rarr; <i>reviewer</i> role</li>
</ul>
<p>This analyzer is automatically registered in the catalog on
server startup if not already present.</p>

<h3>Remote analyzers</h3>
<p>A remote analyzer (e.g. <b>AuraXLM</b>) is an external MCP
service that takes over routing decisions.  When a remote analyzer
is active, AuraRouter delegates the classify-and-dispatch step to it
via an MCP callback.  If the remote analyzer fails or is unreachable,
AuraRouter falls back to the built-in analyzer automatically.</p>

<h3>Switching analyzers</h3>
<p>The active analyzer is stored in
<code>system.active_analyzer</code> in your config.  You can
change it via:</p>
<ul>
  <li>CLI: <code>aurarouter analyzer set &lt;id&gt;</code></li>
  <li>MCP tool: <code>aurarouter.analyzer.set_active</code></li>
  <li>GUI: the <b>Active Analyzer</b> indicator in the Routing
      panel</li>
</ul>
<p>To revert to the built-in default, use
<code>aurarouter analyzer clear</code>.</p>
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
              "configuration", "analyzer", "active"],
    related=["concept.fallback", "concept.roles", "concept.analyzers",
             "panel.settings"],
    body="""\
<h2>Routing Editor</h2>
<p>The <b>Routing</b> section of the Configuration tab lets you
manage fallback chains for each role.</p>

<h3>Active Analyzer</h3>
<p>At the top of the Routing section, the <b>Active Analyzer</b>
indicator shows which route analyzer is currently controlling task
dispatch.  By default this is <code>aurarouter-default</code> (the
built-in intent-triage analyzer).  You can switch to a remote
analyzer via the CLI (<code>aurarouter analyzer set &lt;id&gt;</code>)
or MCP tools.</p>

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
              "manage", "browse", "artifact", "service", "analyzer"],
    related=["concept.providers", "concept.catalog", "concept.analyzers",
             "howto.first_model"],
    body="""\
<h2>Models Tab</h2>
<p>The Models tab lets you browse, download, and manage artifacts from
the <b>Unified Artifact Catalog</b>.</p>

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

<h3>Services and Analyzers</h3>
<p>The catalog now also shows <b>service</b> and <b>analyzer</b>
artifacts alongside models.  Services are external MCP endpoints;
analyzers control how routing decisions are made.  Use the
<i>Kind</i> filter dropdown to narrow the view to a single artifact
type.</p>

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

HELP.register(HelpTopic(
    id="howto.analyzer",
    title="How to Manage Route Analyzers",
    category="howto",
    keywords=["analyzer", "route", "switch", "active", "built-in", "remote",
              "auraxlm", "triage"],
    related=["concept.analyzers", "concept.catalog", "panel.routing"],
    body="""\
<h2>Manage Route Analyzers</h2>
<p>Route analyzers control how AuraRouter classifies and dispatches
tasks.  This guide covers viewing, switching, and connecting
analyzers.</p>

<h3>View the Active Analyzer</h3>
<p>From the CLI:</p>
<pre>aurarouter analyzer active</pre>
<p>Or via MCP: call <code>aurarouter.analyzer.get_active()</code>.
The GUI shows the active analyzer at the top of the Routing
panel.</p>

<h3>List Available Analyzers</h3>
<pre>aurarouter analyzer list</pre>
<p>This lists all artifacts of kind <code>analyzer</code> in the
unified catalog.  The active analyzer is marked with an asterisk.</p>

<h3>Switch Analyzers</h3>
<pre>aurarouter analyzer set auraxlm-remote</pre>
<p>This sets the active analyzer to <code>auraxlm-remote</code>
(or any other analyzer ID registered in the catalog).  Save is
automatic.</p>

<h3>Revert to the Built-in Analyzer</h3>
<pre>aurarouter analyzer clear</pre>
<p>This clears the <code>system.active_analyzer</code> setting.
AuraRouter will use its built-in <code>aurarouter-default</code>
analyzer, which performs intent classification with
complexity-based triage routing.</p>

<h3>What the Built-in Analyzer Does</h3>
<p>The <code>aurarouter-default</code> analyzer classifies each
task by intent (e.g. <code>SIMPLE_CODE</code>,
<code>COMPLEX_REASONING</code>) and assigns a complexity score
from 1 to 10.  Based on these, it routes to the appropriate role
chain.  Simple tasks go straight to execution; complex tasks go
through the full Plan &rarr; Execute &rarr; Review pipeline.</p>

<h3>Connect a Remote Analyzer (AuraXLM)</h3>
<p>To use a remote analyzer such as AuraXLM:</p>
<ol>
  <li>Register the analyzer in the catalog:
<pre>aurarouter catalog register auraxlm-remote \\
  --kind analyzer \\
  --display-name "AuraXLM Remote Analyzer"</pre></li>
  <li>Add MCP endpoint details in <code>auraconfig.yaml</code>:
<pre>
catalog:
  auraxlm-remote:
    kind: analyzer
    display_name: AuraXLM Remote Analyzer
    mcp_endpoint: http://auraxlm-host:9090
    mcp_tool_name: auraxlm.analyze
</pre></li>
  <li>Activate it:
      <pre>aurarouter analyzer set auraxlm-remote</pre></li>
</ol>
<p>If the remote analyzer becomes unreachable, AuraRouter
automatically falls back to the built-in analyzer so tasks still
get processed.</p>
""",
))

HELP.register(HelpTopic(
    id="custom-intents",
    title="Custom Domain-Specific Intents",
    category="concept",
    keywords=["intent", "custom", "domain", "role_bindings", "analyzer",
              "classification", "registry"],
    related=["concept.analyzers", "concept.triage", "intent-selection",
             "analyzer-intents", "routing-advisors"],
    body="""\
<h2>Custom Domain-Specific Intents</h2>
<p>AuraRouter classifies every incoming task into an <b>intent</b>
that determines which role (and therefore which model chain) handles
execution.  Three built-in intents are always available:</p>

<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Intent</th><th>Target Role</th><th>Description</th></tr>
<tr><td><b>DIRECT</b></td><td>coding</td>
    <td>Simple questions or single-turn tasks</td></tr>
<tr><td><b>SIMPLE_CODE</b></td><td>coding</td>
    <td>Straightforward code generation</td></tr>
<tr><td><b>COMPLEX_REASONING</b></td><td>reasoning</td>
    <td>Multi-step reasoning or architectural design</td></tr>
</table>

<h3>Declaring Custom Intents</h3>
<p>Analyzers can declare additional intents through the
<code>role_bindings</code> field in their catalog spec.  Each key
becomes a custom intent name and its value is the target role:</p>
<pre>
role_bindings:
  sar_detection: coding
  sar_geolocation: reasoning
  explain_result: reasoning
</pre>

<p>Custom intents are registered in the <b>IntentRegistry</b> when
the analyzer is activated.  They have higher priority than built-in
intents, so a custom intent with the same name as a built-in intent
will override it.</p>

<h3>How It Works</h3>
<ol>
  <li>The active analyzer&rsquo;s <code>role_bindings</code> are read
      from the catalog.</li>
  <li>Each binding is converted into an <code>IntentDefinition</code>
      and registered in the <code>IntentRegistry</code>.</li>
  <li>During task routing, the router model classifies the task into
      one of the available intents (built-in + custom).</li>
  <li>The intent&rsquo;s target role determines which model chain
      executes the task.</li>
</ol>

<h3>Model Eligibility</h3>
<p>Models can declare <code>supported_intents</code> in their catalog
entry to indicate which intents they handle best.  When set,
<code>filter_chain_by_intent()</code> narrows the model chain to
prefer models that explicitly support the classified intent.</p>
""",
))

HELP.register(HelpTopic(
    id="intent-selection",
    title="How to Select an Intent",
    category="howto",
    keywords=["intent", "select", "combobox", "override", "force",
              "classify", "auto", "manual"],
    related=["custom-intents", "panel.workspace", "concept.triage",
             "analyzer-intents"],
    body="""\
<h2>Select a Specific Intent</h2>
<p>By default, AuraRouter auto-classifies every task into the most
appropriate intent.  You can override this by selecting a specific
intent before submitting.</p>

<h3>GUI: Intent Combobox</h3>
<p>On the <b>Workspace</b> panel, the intent combobox sits next to
the <b>Execute</b> button.  It shows:</p>
<ul>
  <li><b>Auto (classify)</b> &mdash; default; lets the router model
      decide.</li>
  <li><b>Built-in</b> group &mdash; DIRECT, SIMPLE_CODE,
      COMPLEX_REASONING.</li>
  <li><b>Analyzer</b> group &mdash; custom intents from the active
      analyzer (if any).  The group is labeled with the analyzer&rsquo;s
      display name.</li>
</ul>
<p>Select a specific intent, then click <b>Execute</b> (or press
<b>Ctrl+Enter</b>).  The classifier step is skipped and the task
routes directly to the intent&rsquo;s target role.</p>

<p>The combobox refreshes automatically when the active analyzer
changes (e.g., via the Settings panel).</p>

<h3>CLI: --intent Flag</h3>
<p>Use the <code>--intent</code> flag with <code>aurarouter run</code>
to force a specific intent from the command line:</p>
<pre>aurarouter run "Detect targets in SAR scene" --intent sar_detection</pre>

<p>If the intent name is not recognized, the CLI prints an error with
the list of available intents.</p>

<h3>MCP: route_task intent Parameter</h3>
<p>The <code>route_task</code> MCP tool accepts an optional
<code>intent</code> parameter.  When provided, auto-classification is
bypassed.</p>

<h3>Discovering Available Intents</h3>
<p>Use the CLI to see all available intents:</p>
<pre>
aurarouter intent list
aurarouter intent describe SIMPLE_CODE
</pre>
<p>Or call the <code>list_intents</code> MCP tool for a JSON
response.</p>
""",
))

HELP.register(HelpTopic(
    id="analyzer-intents",
    title="Analyzer Intent Reference",
    category="concept",
    keywords=["analyzer", "intent", "reference", "role_bindings",
              "auracode", "auraxlm", "contract", "spec"],
    related=["custom-intents", "concept.analyzers", "intent-selection",
             "routing-advisors"],
    body="""\
<h2>Analyzer Intent Reference</h2>
<p>This topic lists the analyzer spec fields related to intent
declaration and the reference contracts shipped with AuraRouter.</p>

<h3>Analyzer Spec Fields</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Field</th><th>Type</th><th>Required</th><th>Description</th></tr>
<tr><td><code>analyzer_kind</code></td><td>string</td><td>Yes</td>
    <td>Analyzer type: <code>intent_triage</code>,
    <code>moe_ranking</code>, or custom.</td></tr>
<tr><td><code>role_bindings</code></td><td>dict</td><td>No</td>
    <td>Maps intent names to target roles.  Each key becomes a
    registered intent.</td></tr>
<tr><td><code>mcp_endpoint</code></td><td>URL string</td><td>No</td>
    <td>MCP endpoint for remote analyzers.</td></tr>
<tr><td><code>mcp_tool_name</code></td><td>string</td><td>No</td>
    <td>Tool name to call on the remote endpoint.</td></tr>
<tr><td><code>capabilities</code></td><td>list</td><td>No</td>
    <td>Declared capabilities for catalog filtering.</td></tr>
</table>

<h3>Validation</h3>
<p>The <code>validate_analyzer_spec()</code> function checks that
<code>analyzer_kind</code> is present, <code>role_bindings</code>
keys are valid identifiers, binding targets reference configured roles,
and <code>mcp_endpoint</code> is a well-formed URL.  Validation is
warn-only for backwards compatibility.</p>

<h3>AuraCode Contract</h3>
<p>The <code>contracts/auracode.py</code> module defines intents for
code-focused workflows:</p>
<pre>
generate_code  &rarr; coding
edit_code      &rarr; coding
complete_code  &rarr; coding
explain_code   &rarr; reasoning
review         &rarr; reasoning
chat           &rarr; reasoning
plan           &rarr; reasoning
</pre>
<p>Use <code>create_auracode_analyzer_spec()</code> to get a ready-made
spec dict for an AuraCode-compatible analyzer.</p>

<h3>AuraXLM Contract</h3>
<p>The <code>contracts/auraxlm.py</code> module defines the MoE ranking
analyzer interface.  It uses <code>analyzer_kind: moe_ranking</code>
and calls the <code>auraxlm.analyze_route</code> tool on the remote
endpoint, passing the prompt, optional intent hint, candidate models,
cost/latency constraints, and returning a ranked model list with
role recommendation.</p>
""",
))

HELP.register(HelpTopic(
    id="routing-advisors",
    title="Routing Advisors",
    category="concept",
    keywords=["advisor", "routing", "chain", "reorder", "mcp",
              "service", "fabric"],
    related=["concept.analyzers", "custom-intents", "analyzer-intents",
             "concept.fallback"],
    body="""\
<h2>Routing Advisors</h2>
<p><b>Routing advisors</b> are MCP services that can reorder a
role&rsquo;s model chain before execution.  They sit between intent
classification and model execution, providing an additional layer
of routing intelligence.</p>

<h3>How They Work</h3>
<ol>
  <li>After intent classification determines the target role,
      <code>ComputeFabric.consult_routing_advisors()</code> sends
      the role, current chain, and classified intent to each
      registered advisor.</li>
  <li>Advisors with the <code>chain_reorder</code> capability can
      return a reordered chain (e.g., promoting a model that performs
      well for the given intent).</li>
  <li>If no advisor responds or all fail, the original chain is
      used unchanged.</li>
</ol>

<h3>Registration</h3>
<p>Advisors can be registered in two ways:</p>
<ul>
  <li><b>Programmatic:</b>
      <code>fabric.register_routing_advisor(client)</code> &mdash;
      registers an MCP client as an advisor.  Idempotent.</li>
  <li><b>Catalog auto-discovery:</b> declare a service artifact with
      the <code>routing_advisor</code> capability:
<pre>
catalog:
  my-advisor:
    kind: service
    capabilities: [routing_advisor]
    endpoint: http://advisor-host:9090
</pre>
      These are auto-registered on startup.</li>
</ul>

<h3>Management</h3>
<table border="1" cellpadding="4" cellspacing="0">
<tr><th>Method</th><th>Description</th></tr>
<tr><td><code>register_routing_advisor(client)</code></td>
    <td>Add an advisor (idempotent)</td></tr>
<tr><td><code>unregister_routing_advisor(id)</code></td>
    <td>Remove an advisor by ID</td></tr>
<tr><td><code>list_routing_advisors()</code></td>
    <td>List all registered advisor IDs</td></tr>
<tr><td><code>consult_routing_advisors(role, chain, intent)</code></td>
    <td>Query advisors for chain reordering</td></tr>
</table>

<p>Routing advisors are intent-aware &mdash; the classified intent
is passed to each advisor so it can make intent-specific reordering
decisions.  This is especially useful for domain-specific intents
where certain models are known to perform better.</p>
""",
))


# ==================================================================
# TG7 additions: new concept, panel, and howto topics
# ==================================================================

HELP.register(HelpTopic(
    id="concept.speculative",
    title="Speculative Decoding",
    category="concept",
    keywords=["speculative", "decoding", "drafter", "verifier", "draft",
              "notional", "acceptance"],
    related=["concept.notional", "panel.monitor"],
    body="""\
<h2>Speculative Decoding</h2>
<p>Speculative decoding uses a fast <em>drafter</em> model to generate a candidate
response at high speed, then a stronger <em>verifier</em> model validates it before
delivery. Accepted drafts are delivered immediately; rejected drafts trigger a
correction pass.</p>
<p>AuraRouter activates speculative decoding automatically when task complexity
reaches the configured threshold (default: 7/10). The notional response protocol
sends draft tokens to the GUI while verification is in progress &mdash; you see output
almost instantly, with a green shield icon confirming verification.</p>
<p>Key metrics to watch in Monitor &rarr; Speculative tab: <em>acceptance rate</em>
(higher = drafter quality is good) and <em>notional emitted</em> (count of
pre-verified responses delivered).</p>
<p>Configure thresholds in Settings &rarr; Speculative Decoding.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.monologue",
    title="AuraMonologue Reasoning",
    category="concept",
    keywords=["monologue", "reasoning", "generator", "critic", "refiner",
              "multi-expert", "convergence"],
    related=["concept.speculative", "panel.monitor"],
    body="""\
<h2>AuraMonologue Reasoning</h2>
<p>Monologue reasoning decomposes complex tasks across three expert roles running
in sequence: a <em>Generator</em> (&amp;#x270F;) produces an initial response; a
<em>Critic</em> (&amp;#x1F50D;) evaluates it against quality criteria; a
<em>Refiner</em> (&amp;#x21BB;) incorporates the critique to produce an improved output.
This loop repeats until the output <em>converges</em> (confidence score exceeds
the threshold) or the maximum iteration limit is reached.</p>
<p>MAS (Multi-Agent Scoring) gates each step &mdash; if an expert's contribution
scores below the relevancy threshold, that step is idled and the loop
continues without it.</p>
<p>Enable monologue in Settings &rarr; Monologue Reasoning. View traces in
Monitor &rarr; Monologue tab.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.sovereignty",
    title="Sovereignty Enforcement",
    category="concept",
    keywords=["sovereignty", "gate", "pii", "cloud", "block", "restrict",
              "open", "local"],
    related=["concept.privacy", "howto.sovereignty_patterns"],
    body="""\
<h2>Sovereignty Enforcement</h2>
<p>The Sovereignty Gate evaluates every prompt before routing. It returns one
of three verdicts:</p>
<ul>
<li><strong>OPEN</strong> &mdash; No restrictions. Prompt can route to any model.</li>
<li><strong>SOVEREIGN</strong> &mdash; Sensitive data detected. Route to local models only.</li>
<li><strong>BLOCKED</strong> &mdash; Content violates policy. Request rejected entirely.</li>
</ul>
<p>Sovereignty checks PII patterns (SSN, credit card numbers, email addresses),
custom patterns you define (FOUO markers, ITAR notices, internal-only labels),
and the PrivacyAuditor's severity rules.</p>
<p>Manage patterns in Routing &rarr; Sovereignty Gate. Test prompts with the dry-run
evaluator. View enforcement statistics in Monitor &rarr; Privacy tab.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.sessions",
    title="Multi-Turn Sessions",
    category="concept",
    keywords=["session", "multi-turn", "conversation", "context", "history",
              "gist", "condense"],
    related=["howto.session", "concept.speculative"],
    body="""\
<h2>Multi-Turn Sessions</h2>
<p>Sessions let you have multi-turn conversations with your models. AuraRouter
maintains message history, tracks context pressure, and automatically condenses
older messages into <em>gists</em> when the context window fills up.</p>
<p>Enable sessions with the Session Mode toggle in the Workspace panel. Create
a new session with the + button. Each session shows a <em>token pressure gauge</em>
&mdash; a horizontal bar that turns red as the context window fills. At 95%+, you'll
see a "Click to Condense Now" button for manual gisting.</p>
<p>Sessions persist across restarts via SQLite storage. You can list and resume
previous sessions from the left sidebar.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.notional",
    title="Notional Responses",
    category="concept",
    keywords=["notional", "response", "draft", "stream", "confidence",
              "pre-verified", "latency"],
    related=["concept.speculative", "concept.sessions"],
    body="""\
<h2>Notional Responses</h2>
<p>A notional response is a draft output emitted by the speculative decoder
<em>before</em> the verifier model has confirmed it. When the drafter's
confidence score exceeds the notional confidence threshold (default: 0.85),
AuraRouter streams the draft tokens to the UI immediately.</p>
<p>If the verifier accepts the draft, the response is marked with a green
shield (&amp;#x2713; Verified). If the verifier rejects it, a correction event fires
and the response is automatically updated with the verified version.</p>
<p>Notional responses dramatically reduce perceived latency for high-confidence
simple tasks while maintaining the quality guarantee of the verifier model.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.rag",
    title="RAG Enrichment",
    category="concept",
    keywords=["rag", "retrieval", "augmentation", "auraxlm", "knowledge",
              "enrichment", "semantic"],
    related=["concept.analyzers", "concept.sessions"],
    body="""\
<h2>RAG Enrichment</h2>
<p>When RAG enrichment is enabled, AuraRouter queries the AuraXLM knowledge
base before executing each task. Relevant snippets are injected into the prompt
context, giving the model access to your organization's domain knowledge without
fine-tuning.</p>
<p>The enrichment pipeline searches by semantic similarity, respects a configurable
token budget (default: 2048 tokens), and falls back gracefully if AuraXLM is
unavailable. Retrieval latency is tracked in Monitor &rarr; ROI tab.</p>
<p>Configure in Settings &rarr; RAG Enrichment. Requires AuraXLM endpoint in config.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.intents",
    title="Intent Classification",
    category="concept",
    keywords=["intent", "classification", "registry", "role", "binding",
              "zero-shot", "triage"],
    related=["concept.analyzers", "howto.custom_intent"],
    body="""\
<h2>Intent Classification</h2>
<p>Every task that enters AuraRouter is classified into an <em>intent</em> &mdash;
a named category that determines which routing role handles it. Built-in intents
include DIRECT, SIMPLE_CODE, and COMPLEX_REASONING.</p>
<p>Analyzers can declare additional intents through their <code>role_bindings</code>
configuration. When two sources declare the same intent, the higher-priority source
wins (analyzer intents default to priority 10 vs. builtin priority 0).</p>
<p>The intent classifier is a zero-shot LLM call that picks the best matching
intent from the registry. The result feeds into the Triage Router to select
the final model chain.</p>
<p>Browse and manage intents in Routing &rarr; Intent Registry. Test routing in
Routing &rarr; Route Simulator.</p>
""",
))

HELP.register(HelpTopic(
    id="concept.feedback",
    title="Feedback-Driven Triage",
    category="concept",
    keywords=["feedback", "triage", "ema", "success", "rate", "complexity",
              "adaptive", "learning"],
    related=["howto.model_performance", "concept.triage"],
    body="""\
<h2>Feedback-Driven Triage</h2>
<p>The Triage Router uses complexity scores to assign routing roles. Over time,
it tracks success rates per model per complexity band via the Feedback Store.
When a model consistently underperforms on certain complexity ranges, the triage
thresholds shift using an exponential moving average (EMA).</p>
<p>This means AuraRouter learns from your specific workload &mdash; a model that
handles moderate complexity well will gradually receive more of those tasks
without manual reconfiguration.</p>
<p>View current triage rules in Routing &rarr; Triage Rules. View per-model success
rates in Monitor &rarr; Performance tab.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.workspace",
    title="Workspace Panel",
    category="panel",
    keywords=["workspace", "workspace panel", "session", "single-shot",
              "routing pill", "token pressure", "execute"],
    related=["howto.session", "concept.sessions"],
    body="""\
<h2>Workspace Panel</h2>
<p>The Workspace is where you execute tasks and have conversations with your
models. It has two modes:</p>
<h3>Single-Shot Mode (default)</h3>
<p>Type a task, press Ctrl+Return to execute. The output area shows results
with progressive rendering as tokens arrive. The Review Loop section below
the output shows pass/fail verdicts and correction diffs when the review system
triggers.</p>
<h3>Session Mode</h3>
<p>Toggle "Session Mode" in the header. A chat-like interface replaces the
output area. The left sidebar shows your sessions; the center shows
conversation bubbles with routing insight pills. The token pressure gauge
in the header tracks context window usage.</p>
<h3>Routing Insight Pills</h3>
<p>Every assistant response includes a one-line routing summary pill:
<em>[Local &middot; SIMPLE_CODE &middot; 0.92 conf &middot; $0.02 saved]</em>. Click it for
the full routing decision popover: intent, complexity, strategy, confidence,
cost savings, and execution mode.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.routing",
    title="Routing Panel",
    category="panel",
    keywords=["routing", "pipeline", "intent", "registry", "simulator",
              "sovereignty", "triage", "flowchart"],
    related=["howto.sovereignty_patterns", "howto.custom_intent"],
    body="""\
<h2>Routing Panel</h2>
<p>The Routing Panel shows the complete path from prompt to model, across
four stages:</p>
<ol>
<li><strong>Stage 1 &mdash; Pre-filter:</strong> ONNX-based complexity scoring (priority &ge;50)</li>
<li><strong>Stage 2 &mdash; Intent Classifier:</strong> Assigns an intent from the registry (priority &lt;50)</li>
<li><strong>Sovereignty Gate:</strong> Blocks or restricts cloud routing for sensitive data</li>
<li><strong>Triage Router:</strong> Maps complexity + intent to a role chain</li>
</ol>
<h3>Intent Registry</h3>
<p>Browse all registered intents, see which analyzer declared them, and add
custom intents. Change the active analyzer from the dropdown.</p>
<h3>Route Simulator</h3>
<p>Paste any prompt and click Simulate to watch it traverse the pipeline
stage by stage. Use "Promote to Rule" to save promising simulation results
as triage rules without editing YAML.</p>
<h3>Sovereignty Gate</h3>
<p>Add and test custom PII/classification patterns. The dry-run evaluator
shows verdicts instantly.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.monitor",
    title="Monitor Panel",
    category="panel",
    keywords=["monitor", "traffic", "privacy", "health", "roi", "speculative",
              "monologue", "performance", "observability", "dashboard"],
    related=["concept.speculative", "concept.monologue"],
    body="""\
<h2>Monitor Panel</h2>
<p>The Monitor panel provides full observability into AuraRouter's operation
across 8 tabs:</p>
<ul>
<li><strong>Overview:</strong> Key metrics at a glance</li>
<li><strong>Traffic:</strong> Token counts, spend, model usage</li>
<li><strong>Privacy:</strong> PII events, severity breakdown</li>
<li><strong>Health:</strong> Model provider connectivity</li>
<li><strong>ROI:</strong> Cost avoided, savings sparkline, export report</li>
<li><strong>Speculative:</strong> Draft/verify sessions, acceptance rates</li>
<li><strong>Monologue:</strong> Reasoning traces with expert role coloring</li>
<li><strong>Performance:</strong> Success rate &times; complexity heatmap per model</li>
</ul>
<p>All tabs share the time range controls at the top. Data auto-refreshes
every 30 seconds.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.settings",
    title="Settings Panel",
    category="panel",
    keywords=["settings", "configuration", "speculative", "monologue", "rag",
              "sovereignty", "session", "budget", "yaml", "persona"],
    related=["howto.session", "concept.sessions"],
    body="""\
<h2>Settings Panel</h2>
<p>The Settings panel consolidates all AuraRouter configuration into
collapsible sections:</p>
<ul>
<li><strong>Route Analyzer:</strong> Active analyzer selection and management</li>
<li><strong>MCP Tools:</strong> Enable/disable individual MCP tool endpoints</li>
<li><strong>Budget &amp; Cost:</strong> Daily/monthly spend limits</li>
<li><strong>Privacy:</strong> PII detection rules and severity thresholds</li>
<li><strong>System:</strong> Environment, logging, session settings</li>
<li><strong>Speculative Decoding:</strong> Complexity threshold, confidence gate</li>
<li><strong>Monologue Reasoning:</strong> Max iterations, convergence threshold</li>
<li><strong>Sovereignty Enforcement:</strong> Pattern management link</li>
<li><strong>Session Management:</strong> Condensation threshold, auto-gist</li>
<li><strong>RAG Enrichment:</strong> Max tokens, endpoint timeout</li>
<li><strong>YAML Preview / Editor:</strong> View or edit raw config</li>
</ul>
<p>If you used the onboarding wizard with a persona preset, fields set by
that persona show a small badge (&amp;#x26A1; Performance, &amp;#x1F512; Privacy,
&amp;#x1F52C; Researcher). The badge clears when you override a value.</p>
""",
))

HELP.register(HelpTopic(
    id="panel.session",
    title="Session Chat",
    category="panel",
    keywords=["session", "chat", "multi-turn", "token pressure", "gist",
              "condense", "routing pill"],
    related=["concept.sessions", "howto.session"],
    body="""\
<h2>Session Chat</h2>
<p>Session Chat is activated by the Session Mode toggle in the Workspace
header. It transforms the output area into a persistent conversation.</p>
<h3>Creating Sessions</h3>
<p>Click "&plus; New Session" in the header or let the first send automatically
create one. Sessions appear in the left sidebar with message count.</p>
<h3>Context Pressure</h3>
<p>The token pressure gauge at the top shows how full the context window is.
Zones: green (safe, &lt;60%), yellow (caution, 60-80%), red (pressure, 80-95%),
flashing red (critical, &gt;95%). At critical, click "Click to Condense Now"
to trigger manual gisting.</p>
<h3>Routing Pills</h3>
<p>Each assistant message shows a routing insight pill above the content.
Click it to see the full routing decision for that message.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.session",
    title="How to Use Multi-Turn Sessions",
    category="howto",
    keywords=["session", "multi-turn", "chat", "condense", "gist",
              "token pressure", "guide"],
    related=["concept.sessions", "panel.session"],
    body="""\
<h2>How to Use Multi-Turn Sessions</h2>
<ol>
<li>Open the <strong>Workspace</strong> panel.</li>
<li>Click the <strong>Session Mode</strong> checkbox in the header to enable it.</li>
<li>Click <strong>&plus; New Session</strong> to create a session (or type your first
message and a session is created automatically).</li>
<li>Type messages in the input box and press <strong>Ctrl+Return</strong> to send.</li>
<li>Watch the <strong>token pressure gauge</strong> as the conversation grows.
At 95%+, click "Click to Condense Now" to gist older messages and free up
context space.</li>
<li>Switch sessions from the left sidebar. Previous sessions are loaded with
their full history.</li>
<li>Toggle <strong>Session Mode off</strong> to return to single-shot mode.
Your sessions are preserved.</li>
</ol>
""",
))

HELP.register(HelpTopic(
    id="howto.sovereignty_patterns",
    title="How to Create Sovereignty Patterns",
    category="howto",
    keywords=["sovereignty", "pattern", "pii", "regex", "dry-run",
              "classification", "custom", "gate"],
    related=["concept.sovereignty", "panel.routing"],
    body="""\
<h2>How to Create Sovereignty Patterns</h2>
<ol>
<li>Open the <strong>Routing</strong> panel.</li>
<li>Expand the <strong>Sovereignty Gate</strong> section.</li>
<li>Click <strong>&plus; Add Pattern</strong>.</li>
<li>Enter the pattern details:
  <ul>
    <li><strong>Pattern text:</strong> Regex or keyword to match</li>
    <li><strong>Severity:</strong> low / medium / high</li>
    <li><strong>Description:</strong> What this pattern protects</li>
  </ul>
</li>
<li>Click <strong>Save</strong>. The pattern is active immediately.</li>
<li>Test the pattern using the <strong>dry-run evaluator</strong>: paste a
sample prompt and click Evaluate to see the verdict.</li>
</ol>
<p><strong>Severity guide:</strong></p>
<ul>
<li><strong>High:</strong> Immediate SOVEREIGN verdict &mdash; routes local only</li>
<li><strong>Medium:</strong> SOVEREIGN verdict if multiple matches</li>
<li><strong>Low:</strong> Logged but does not change routing</li>
</ul>
""",
))

HELP.register(HelpTopic(
    id="howto.custom_intent",
    title="How to Create Custom Intents",
    category="howto",
    keywords=["intent", "custom", "registry", "role_bindings", "analyzer",
              "domain", "add"],
    related=["concept.intents", "concept.analyzers"],
    body="""\
<h2>How to Create Custom Intents</h2>
<h3>Via the Intent Registry Editor (easiest)</h3>
<ol>
<li>Open the <strong>Routing</strong> panel.</li>
<li>Expand the <strong>Intent Registry</strong> section.</li>
<li>Click <strong>&plus; Add Custom Intent</strong>.</li>
<li>Fill in the name, description, and target role.</li>
<li>Click Save. The intent is active for the current session.</li>
</ol>
<h3>Via Analyzer role_bindings (persistent)</h3>
<ol>
<li>Select your analyzer in the dropdown and click <strong>Edit Spec</strong>.</li>
<li>Add your intent to the <code>role_bindings</code> section:
<pre>role_bindings:
  my_intent: reasoning</pre>
</li>
<li>Click <strong>Validate</strong> then Save.</li>
</ol>
""",
))

HELP.register(HelpTopic(
    id="howto.speculative_tuning",
    title="How to Tune Speculative Decoding",
    category="howto",
    keywords=["speculative", "tuning", "threshold", "acceptance", "drafter",
              "verifier", "confidence", "latency"],
    related=["concept.speculative", "concept.notional"],
    body="""\
<h2>How to Tune Speculative Decoding</h2>
<p>Speculative decoding has two main thresholds to tune:</p>
<h3>Complexity Threshold (default: 7)</h3>
<p>Tasks below this complexity score use speculative decoding. Higher = fewer
speculative executions (only the most complex tasks). Lower = more speculative
executions (even simple tasks get a draft/verify pass).</p>
<p>Start at 7 and lower it if your acceptance rate is consistently above 90%.</p>
<h3>Notional Confidence Threshold (default: 0.85)</h3>
<p>Only draft responses with confidence above this value are streamed to the UI
before verification. Higher = fewer notional responses sent (safer).
Lower = more notional responses (faster perceived latency, more corrections).</p>
<h3>Reading the Monitor</h3>
<p>Go to Monitor &rarr; Speculative tab. Watch <strong>acceptance rate</strong>:
if it drops below 70%, your drafter model may not be strong enough for those
complexity bands. Consider raising the complexity threshold or upgrading
the drafter.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.model_performance",
    title="How to Read Model Performance Data",
    category="howto",
    keywords=["performance", "heatmap", "success", "rate", "complexity",
              "band", "triage", "model"],
    related=["concept.feedback", "panel.monitor"],
    body="""\
<h2>How to Read Model Performance Data</h2>
<p>Go to Monitor &rarr; Performance tab. The heatmap shows each model's success
rate across complexity bands.</p>
<h3>Color coding</h3>
<ul>
<li><strong>Green (&ge;80%):</strong> Model performs well in this band</li>
<li><strong>Yellow (60-79%):</strong> Acceptable but watch for degradation</li>
<li><strong>Red (&lt;60%):</strong> Consider replacing this model for this band</li>
<li><strong>&mdash; (no data):</strong> Model hasn't been used in this band yet</li>
</ul>
<h3>When to change triage rules</h3>
<p>If a model consistently shows red in complexity band 4-6 but green in 1-3,
edit the triage rule to reduce its max complexity to 3. Use the Route Simulator
to test the new assignment before committing.</p>
""",
))

HELP.register(HelpTopic(
    id="howto.cross_panel_nav",
    title="Navigating Between Panels",
    category="howto",
    keywords=["navigation", "keyboard", "shortcut", "panel", "link",
              "session", "f1"],
    related=["panel.workspace", "panel.settings"],
    body="""\
<h2>Navigating Between Panels</h2>
<h3>Keyboard Shortcuts</h3>
<ul>
<li><strong>F1:</strong> Open Help panel</li>
<li><strong>Ctrl+,:</strong> Open Settings panel</li>
<li><strong>Ctrl+Return:</strong> Execute task / Send message</li>
<li><strong>Ctrl+N:</strong> New task / New session</li>
<li><strong>Escape:</strong> Cancel execution</li>
</ul>
<h3>Click-to-Navigate Links</h3>
<p>Several elements navigate directly to related panels:</p>
<ul>
<li>Click a <strong>model name</strong> in Monitor &rarr; Performance &rarr; opens Models panel</li>
<li>Click a <strong>role name</strong> in Monitor &rarr; navigates to Routing panel</li>
<li>Click a model <strong>node in the flowchart</strong> &rarr; opens Models panel</li>
<li>After execution: <strong>"View in Monitor"</strong> link opens the relevant Monitor tab</li>
<li>Settings &rarr; Sovereignty &rarr; <strong>"Manage &rarr;"</strong> link &rarr; opens Routing panel</li>
</ul>
<h3>Session Indicator</h3>
<p>When session mode is active, the toolbar shows the current session ID and
context usage. This is visible from any panel.</p>
""",
))
