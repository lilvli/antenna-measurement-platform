export type PageId = 'run' | 'history' | 'compensation' | 'devices'
export type RunTopology = 'SOFTWARE_VNA_SWEEP' | 'RTC_STOP_AND_GO' | 'RTC_CONTINUOUS'

export interface ProtocolFieldSummary {
  key: string
  display_name: string
  data_type: string
  unit: string | null
  minimum: number | null
  maximum: number | null
  source: string
  default: unknown
  description: string
  enum_options: Array<{ logical: string; display: string; wire: string }>
}

export interface ProtocolCommandSummary {
  command_id: string
  display_name: string
  auto_role: string
  opcode: number
  opcode_hex: string
  timeout_ms: number
  risk: string
  description: string
  fields: ProtocolFieldSummary[]
}

export interface ProfileSummary {
  asset_id: string
  profile_id: string
  profile_name: string
  protocol_version: string
  supported_polarizations: string
  file_sha256: string
  command_count: number
  commands: ProtocolCommandSummary[]
  capabilities: Record<string, boolean>
  vectors: Array<{ vector_id: string; passed: boolean }>
}

export interface CoordinateSummary {
  asset_id: string
  antenna_id: string
  schema_version: string
  release_status: string
  evidence: 'REAL' | 'SIMULATED'
  evidence_limit: string
  unit: string
  polarizations: string[]
  channel_count: number
  enabled_count: number
  polarization_layouts: Record<string, {
    channel_count: number
    enabled_count: number
    rows: number
    columns: number
  }>
  file_sha256: string
}

export interface SerialPortInfo {
  device: string
  description: string
  hwid: string
  manufacturer: string
}

export interface DeviceRawLog {
  timestamp: string
  device_id: string
  direction: 'TX' | 'RX' | 'ERROR'
  command_id?: string
  raw_hex: string
  transport?: string
  context?: string
  parsed?: {
    profile_id: string
    profile_name: string
    command_id?: string
    fields: Array<{ key: string; label: string; value: unknown; unit?: string | null }>
  }
  error?: string
}

export interface DeviceStatus {
  device_id: 'beam_controller' | 'vna' | 'turntable' | 'rtc'
  source: 'REAL' | 'SIMULATED' | null
  state: string
  identity: string | null
  details: Record<string, unknown>
  updated_at?: string
}

export interface RtcWaveTable {
  entries: Array<{
    address: number
    label: string
    frame_hex: string
    element?: number
    readback_hex?: string
    status: 'PREVIEW' | 'VERIFIED' | 'MATCH' | 'MISMATCH' | 'EMPTY'
  }>
  count: number
  capacity: number
  verified?: boolean
  source?: 'REAL' | 'SIMULATED'
}

export interface RunStatus {
  run_id: string
  state: string
  total: number
  completed: number
  progress: number
  output_path: string | null
  error: Record<string, unknown> | null
  result: Record<string, unknown> | null
  cleanup_pending: boolean
  pause_requested: boolean
  rtc_configuration?: {
    tr: { mode: 'TX' | 'RX'; period_us: number; high_us: number; delay_us: number }
    wave_count?: number
    waves_verified?: boolean
  } | null
  plan: {
    test_type: 'CALIBRATION' | 'PATTERN'
    topology: RunTopology
    beam_control_mode: 'SOFTWARE_DIRECT' | 'EXTERNAL_FIXED'
    signal_path: 'TX' | 'RX'
    polarization: 'H' | 'V'
    s_parameter: string
    profile_id: string
    coordinate_id: string
    array_id: number
    output_directory: string
    base_filename: string
    frequency_start_hz: number
    frequency_stop_hz: number
    frequency_points: number
    if_bandwidth_hz: number
    source_power_dbm: number
    averaging_enabled: boolean
    averaging_count: number
    settle_ms: number
    azimuth_start_deg: number
    azimuth_stop_deg: number
    azimuth_step_deg: number
    elevation_start_deg: number
    elevation_stop_deg: number
    elevation_step_deg: number
    move_speed_deg_s: number
    beams: Array<{
      beam_id: string
      off_axis_deg: number
      azimuth_deg: number
      reference_frequency_hz: number | null
    }>
  }
}

export interface LiveSample {
  run_id: string
  kind: 'CALIBRATION' | 'PATTERN'
  magnitude_db: number | null
  phase_deg?: number | null
  magnitudes_db?: Array<number | null>
  phases_deg?: Array<number | null>
  status?: 'CALIBRATED' | 'SKIPPED_DISABLED' | 'COMPLETE'
  completed?: number
  total?: number
  channel?: { element: number; grid_row: number; grid_column: number }
  point?: { azimuth_deg: number; elevation_deg: number }
  beam?: { beam_id: string; beam_index: number }
}

export interface CurrentRunSnapshot {
  run: RunStatus | null
  samples: LiveSample[]
  control_owner: string | null
  event_sequence: number
}
