/* Shared singletons: state, DOM-element references, polling intervals.
 *
 * Auth: a bearer token is stored in localStorage. The page extracts it
 * from ?token=… on first load and strips it from the URL. On 401, the
 * login overlay shows; password → /admin/api/login → bearer token.
 */

export const TOKEN_KEY = 'llmhub.token';
/* Theme override — same key the pre-paint boot script in index.html reads;
 * absent = follow the OS prefers-color-scheme. */
export const THEME_KEY = 'llmhub.theme';
/* Active tab — persisted by the vendored nav so the installed PWA reopens
 * where you left it (fleet nav contract). */
export const TAB_KEY = 'llmhub.tab';
/* Models tab "Active only" filter — persisted like Plugs' show-hidden
 * toggle in home-automation (#266). */
export const MODELS_ACTIVE_ONLY_KEY = 'llmhub.models.activeOnly';

export const STATUS_POLL_MS = 4000;
export const COUNTERS_POLL_MS = 4000;
export const MODELS_POLL_MS = 5000;
export const STATS_POLL_MS = 2000;
/* Machines tab (issue #309) — only polled while that tab is active. */
export const MACHINES_POLL_MS = 10000;

export const state = {
  tab: 'hub',
  status: null,           // /admin/api/hub/status payload
  models: [],             // /admin/api/models payload
  counters: [],           // rows from /admin/api/hub/counters
  liveRequests: [],       // ring synced from SSE stream
  recentErrors: [],
  logLines: [],
  logPaused: false,
  installRows: [],
  version: null,
  hubStreamCtl: null,     // EventSource abort handles
  hubLogStreamCtl: null,

  // Telemetry tab — its own state slice (issue #4).
  telHealth: null,
  telCounters: [],
  telTraces: [],
  telStreamCtl: null,
  telExpandedTraceId: '',
  // Claude Code (host CLI) OTel-sourced usage panel (issue #68).
  telCcSummary: null,
  telCcPeriod: 'today',   // 'today' | 'week' | 'month' | 'all'

  // Code-usage tab — host Claude Code session data (issue #20).
  cldSummary: null,
  cldPeriod: 'today',   // 'today' | 'week' | 'month' | 'all'
  cldVendor: 'all',     // 'all' | 'claude' | 'codex' | 'copilot' | <agentsview agent> (issues #71, #231, #280)

  // Services card — Docker + Langfuse status (issue #27).
  services: null,
  servicesLaunching: false,
  peerBusyIds: {},   // host_id -> true while that peer's wake/sync is in flight (#181, generalized #372)
  agentsviewBusy: false,   // AgentsView start/stop in flight (#280, #284)
  dockerBusy: false,   // Docker Desktop start/stop in flight (#284)
  langfuseBusy: false,   // Langfuse stack start/stop in flight (#284)

  // Startup card — what autostarts with the hub (issue #265).
  startupProfile: null,   // /admin/api/startup-profile payload

  // Fleet placement card — fleet-wide desired state (issue #354).
  fleetPlacement: null,          // /admin/api/fleet-placement payload
  fleetPlacementState: 'loading',// loading | ready | empty | stale | error
  fleetPlacementUpdated: 0,      // Date.now() of the last good fetch

  // Models tab — "Active only" filter, default on (issue #266).
  modelsActiveOnly: true,
  // Models tab — config-as-code context (#424): models.yaml HEAD sha,
  // whether this hub may write (tower only), and the fleet host list the
  // chain editor offers. Null until the first /admin/api/models fetch.
  modelsConfig: null,

  // Machines tab — fleet status/actions/terminal (issue #309).
  machinesStatus: null,     // last good /admin/api/machines/status payload
  machinesDataState: 'loading',  // loading | ready | empty | stale | error
  machinesLastUpdated: 0,   // Date.now() of the last good fetch
  machinesErrorMsg: '',     // sanitized message for the error empty-state
  machinesBusyIds: {},      // machine id -> true while a reboot/shutdown POST is in flight
  machinesRecheckIds: {},   // machine id -> true while a fresh status re-check is pending

  // Diagnostics drill-in (issue #315)
  diagStatus: null,         // last /admin/api/diagnostics/status payload
  diagDataState: 'loading', // loading | ready | empty | stale | error
  diagRuns: [],             // recent capture runs (newest first)
  diagSelectedRun: null,    // run_id whose summary is open
  diagSummary: null,        // summary digest for the selected run
  diagDrift: null,          // drift-vs-baseline for the selected run
  diagSummaryState: 'ready',// lifecycle for the selected-run panel
};

// ES modules are deferred; document.getElementById is safe at top level.
export const els = {
  // Tab buttons + panes are owned by the vendored nav (_vendored/nav) —
  // it discovers them from the markup; no element handles needed here.

  // Telemetry tab — health strip + leaderboard + live trace feed (issue #4)
  telHealth: document.getElementById('telHealth'),
  telHealthText: document.getElementById('telHealthText'),
  telOtelState: document.getElementById('telOtelState'),
  telHashMode: document.getElementById('telHashMode'),
  telEndpoint: document.getElementById('telEndpoint'),
  telOfflineHint: document.getElementById('telOfflineHint'),
  telOpenLangfuse: document.getElementById('telOpenLangfuse'),
  telSummary: document.getElementById('telSummary'),
  telCountersTable: document.getElementById('telCountersTable'),
  telTracesList: document.getElementById('telTracesList'),
  telTracesBadge: document.getElementById('telTracesBadge'),
  telTracesEmpty: document.getElementById('telTracesEmpty'),
  telCcSummary: document.getElementById('telCcSummary'),
  telCcPeriodSeg: document.getElementById('telCcPeriodSeg'),
  telCcTable: document.getElementById('telCcTable'),
  telCcEmpty: document.getElementById('telCcEmpty'),

  // Hub card — live status indicator lives inside the card header
  // (replaces the old always-on status strip).
  hubLiveStatus: document.getElementById('hubLiveStatus'),
  hubLiveStatusText: document.getElementById('hubLiveStatusText'),
  themeToggleBtn: document.getElementById('themeToggleBtn'),
  hubPid: document.getElementById('hubPid'),
  hubUptime: document.getElementById('hubUptime'),
  hubSparklines: document.getElementById('hubSparklines'),

  // Services card — Docker + Langfuse (issue #27)
  servicesCard: document.getElementById('servicesCard'),
  servicesOverall: document.getElementById('servicesOverall'),
  servicesOverallText: document.getElementById('servicesOverallText'),
  dockerStatus: document.getElementById('dockerStatus'),
  dockerStatusText: document.getElementById('dockerStatusText'),
  dockerDetail: document.getElementById('dockerDetail'),
  langfuseStatus: document.getElementById('langfuseStatus'),
  langfuseStatusText: document.getElementById('langfuseStatusText'),
  langfuseDetail: document.getElementById('langfuseDetail'),
  dockerStartBtn: document.getElementById('dockerStartBtn'),
  dockerStopBtn: document.getElementById('dockerStopBtn'),
  langfuseStartBtn: document.getElementById('langfuseStartBtn'),
  langfuseStopBtn: document.getElementById('langfuseStopBtn'),
  agentsviewStatus: document.getElementById('agentsviewStatus'),
  agentsviewStatusText: document.getElementById('agentsviewStatusText'),
  agentsviewDetail: document.getElementById('agentsviewDetail'),
  agentsviewStartBtn: document.getElementById('agentsviewStartBtn'),
  agentsviewStopBtn: document.getElementById('agentsviewStopBtn'),
  peerRows: document.getElementById('peerRows'),
  servicesActions: document.getElementById('servicesActions'),
  servicesLaunchBtn: document.getElementById('servicesLaunchBtn'),
  servicesHint: document.getElementById('servicesHint'),

  // Health & install
  installCard: document.getElementById('installCard'),
  installSummary: document.getElementById('installSummary'),
  installRows: document.getElementById('installRows'),
  installFixAllBtn: document.getElementById('installFixAllBtn'),
  installRefreshBtn: document.getElementById('installRefreshBtn'),

  // Diagnostic disclosure cards (Live / Counters / Errors / Log — #215)
  liveRequestsList: document.getElementById('liveRequestsList'),
  liveRequestsBadge: document.getElementById('liveRequestsBadge'),
  liveRequestsEmpty: document.getElementById('liveRequestsEmpty'),
  countersTable: document.getElementById('countersTable'),
  recentErrorsList: document.getElementById('recentErrorsList'),
  recentErrorsBadge: document.getElementById('recentErrorsBadge'),
  recentErrorsEmpty: document.getElementById('recentErrorsEmpty'),
  hubLog: document.getElementById('hubLog'),
  hubLogPauseBtn: document.getElementById('hubLogPauseBtn'),

  // Models tab
  modelsList: document.getElementById('modelsList'),
  modelsEmpty: document.getElementById('modelsEmpty'),
  modelsActiveToggle: document.getElementById('modelsActiveToggle'),
  modelsConfigSha: document.getElementById('modelsConfigSha'),

  // Startup card (issue #265)
  startupList: document.getElementById('startupList'),

  // Fleet placement card (issue #354)
  fleetPlacementBody: document.getElementById('fleetPlacementBody'),
  fleetPlacementStaleNote: document.getElementById('fleetPlacementStaleNote'),

  // Model decisions card (issue #373) — Hub tab, role → model + placement
  rolesCard: document.getElementById('rolesCard'),
  rolesStatus: document.getElementById('rolesStatus'),
  rolesList: document.getElementById('rolesList'),
  rolesPlacementList: document.getElementById('rolesPlacementList'),
  rolesViewPlacementBtn: document.getElementById('rolesViewPlacementBtn'),

  // Playground
  playgroundModel: document.getElementById('playgroundModel'),
  playgroundSystem: document.getElementById('playgroundSystem'),
  playgroundPrompt: document.getElementById('playgroundPrompt'),
  playgroundMore: document.getElementById('playgroundMore'),
  playgroundAttachment: document.getElementById('playgroundAttachment'),
  playgroundAttachmentBtn: document.getElementById('playgroundAttachmentBtn'),
  playgroundAttachmentName: document.getElementById('playgroundAttachmentName'),
  playgroundMaxTokens: document.getElementById('playgroundMaxTokens'),
  playgroundMaxTokensSeg: document.getElementById('playgroundMaxTokensSeg'),
  playgroundSendBtn: document.getElementById('playgroundSendBtn'),
  playgroundClearBtn: document.getElementById('playgroundClearBtn'),
  playgroundLatency: document.getElementById('playgroundLatency'),
  playgroundReply: document.getElementById('playgroundReply'),
  playgroundUsage: document.getElementById('playgroundUsage'),
  // Playground — image generation / editing tester (issue #114)
  imageCard: document.getElementById('imageCard'),
  imageModel: document.getElementById('imageModel'),
  imagePrompt: document.getElementById('imagePrompt'),
  imageAttachment: document.getElementById('imageAttachment'),
  imageAttachmentBtn: document.getElementById('imageAttachmentBtn'),
  imageAttachmentName: document.getElementById('imageAttachmentName'),
  imageGenBtn: document.getElementById('imageGenBtn'),
  imageClearBtn: document.getElementById('imageClearBtn'),
  imageLatency: document.getElementById('imageLatency'),
  imagePreview: document.getElementById('imagePreview'),
  imageDownload: document.getElementById('imageDownload'),
  imageDownloadRow: document.getElementById('imageDownloadRow'),
  // Playground — text-to-speech tester (issue #98)
  ttsModel: document.getElementById('ttsModel'),
  ttsAvailability: document.getElementById('ttsAvailability'),
  ttsInput: document.getElementById('ttsInput'),
  ttsLanguage: document.getElementById('ttsLanguage'),
  ttsVoice: document.getElementById('ttsVoice'),
  ttsFormat: document.getElementById('ttsFormat'),
  ttsStreamGroup: document.getElementById('ttsStreamGroup'),
  ttsStream: document.getElementById('ttsStream'),
  ttsSpeedGroup: document.getElementById('ttsSpeedGroup'),
  ttsSpeed: document.getElementById('ttsSpeed'),
  ttsSpeedVal: document.getElementById('ttsSpeedVal'),
  ttsExaggerationGroup: document.getElementById('ttsExaggerationGroup'),
  ttsExaggeration: document.getElementById('ttsExaggeration'),
  ttsExaggerationVal: document.getElementById('ttsExaggerationVal'),
  ttsCfgWeightGroup: document.getElementById('ttsCfgWeightGroup'),
  ttsCfgWeight: document.getElementById('ttsCfgWeight'),
  ttsCfgWeightVal: document.getElementById('ttsCfgWeightVal'),
  ttsSpeakBtn: document.getElementById('ttsSpeakBtn'),
  ttsLatency: document.getElementById('ttsLatency'),
  ttsAudio: document.getElementById('ttsAudio'),
  ttsCard: document.getElementById('ttsCard'),

  // Code-usage tab (issue #20)
  cldFreshness: document.getElementById('cldFreshness'),
  cldVendorSeg: document.getElementById('cldVendorSeg'),
  cldAgentsviewHint: document.getElementById('cldAgentsviewHint'),
  cldPeriodSeg: document.getElementById('cldPeriodSeg'),
  cldRequests: document.getElementById('cldRequests'),
  cldTotalCost: document.getElementById('cldTotalCost'),
  cldInputTok: document.getElementById('cldInputTok'),
  cldOutputTok: document.getElementById('cldOutputTok'),
  cldCacheRead: document.getElementById('cldCacheRead'),
  cldInputCost: document.getElementById('cldInputCost'),
  cldOutputCost: document.getElementById('cldOutputCost'),
  cldOutputReasoning: document.getElementById('cldOutputReasoning'),
  cldCacheCost: document.getElementById('cldCacheCost'),
  cldDeltaRequests: document.getElementById('cldDeltaRequests'),
  cldDeltaInputTok: document.getElementById('cldDeltaInputTok'),
  cldDeltaOutputTok: document.getElementById('cldDeltaOutputTok'),
  cldDeltaCacheRead: document.getElementById('cldDeltaCacheRead'),
  cldModelTable: document.getElementById('cldModelTable'),
  cldModelEmpty: document.getElementById('cldModelEmpty'),
  cldProjectTable: document.getElementById('cldProjectTable'),
  cldProjectEmpty: document.getElementById('cldProjectEmpty'),
  cldVendorCard: document.getElementById('cldVendorCard'),
  cldVendorTable: document.getElementById('cldVendorTable'),
  cldVendorEmpty: document.getElementById('cldVendorEmpty'),
  // Copilot official billing card (issue #231)
  cldCopilotBillingCard: document.getElementById('cldCopilotBillingCard'),
  cldCopilotBillingAsOf: document.getElementById('cldCopilotBillingAsOf'),
  cldCopilotBillingTotal: document.getElementById('cldCopilotBillingTotal'),
  cldCopilotBillingTable: document.getElementById('cldCopilotBillingTable'),
  cldCopilotBillingEmpty: document.getElementById('cldCopilotBillingEmpty'),
  cldCopilotBillingEmptyMsg: document.getElementById('cldCopilotBillingEmptyMsg'),
  // Charts (issue #50)
  cldChartsCard: document.getElementById('cldChartsCard'),
  cldChartInput: document.getElementById('cldChartInput'),
  cldChartOutput: document.getElementById('cldChartOutput'),
  cldChartReqs: document.getElementById('cldChartReqs'),
  cldChartCache: document.getElementById('cldChartCache'),
  cldSessionsList: document.getElementById('cldSessionsList'),
  cldSessionsBadge: document.getElementById('cldSessionsBadge'),
  cldSessionsEmpty: document.getElementById('cldSessionsEmpty'),

  // Machines tab (issue #309)
  machinesLoading: document.getElementById('machinesLoading'),
  machinesError: document.getElementById('machinesError'),
  machinesErrorMsg: document.getElementById('machinesErrorMsg'),
  machinesRetryBtn: document.getElementById('machinesRetryBtn'),
  machinesStaleNote: document.getElementById('machinesStaleNote'),
  machinesList: document.getElementById('machinesList'),
  machinesConfirmDialog: document.getElementById('machinesConfirmDialog'),
  machinesConfirmTitle: document.getElementById('machinesConfirmTitle'),
  machinesConfirmCloseBtn: document.getElementById('machinesConfirmCloseBtn'),
  machinesConfirmBody: document.getElementById('machinesConfirmBody'),
  machinesConfirmBtn: document.getElementById('machinesConfirmBtn'),
  machinesTerminalDialog: document.getElementById('machinesTerminalDialog'),
  machinesTerminalTitle: document.getElementById('machinesTerminalTitle'),
  machinesTerminalCloseBtn: document.getElementById('machinesTerminalCloseBtn'),
  machinesTerminalUnavailable: document.getElementById('machinesTerminalUnavailable'),
  machinesTerminalUnavailableMsg: document.getElementById('machinesTerminalUnavailableMsg'),
  machinesTerminalMount: document.getElementById('machinesTerminalMount'),
  diagDialog: document.getElementById('diagDialog'),
  diagCloseBtn: document.getElementById('diagCloseBtn'),
  diagBody: document.getElementById('diagBody'),

  // Misc
  toast: document.getElementById('toast'),
  loginOverlay: document.getElementById('loginOverlay'),
  loginForm: document.getElementById('loginForm'),
  loginPassword: document.getElementById('loginPassword'),
  loginError: document.getElementById('loginError'),
  buildReadout: document.getElementById('buildReadout'),
};
