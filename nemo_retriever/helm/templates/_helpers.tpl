{{/*
=============================================================================
Naming helpers
=============================================================================
*/}}

{{/*
nemo-retriever.name
  The chart name, optionally overridden by .Values.nameOverride.
*/}}
{{- define "nemo-retriever.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.fullname
  Default fully qualified app name.  Defaults to <release>-<chart> but
  collapses to just <release> when the release name already contains the
  chart name (idiomatic Helm pattern).
*/}}
{{- define "nemo-retriever.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.chart
  Standard Helm chart label value: <name>-<version>, sanitized.
*/}}
{{- define "nemo-retriever.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.serviceAccountName
  Name of the ServiceAccount to use for the service Deployment.
*/}}
{{- define "nemo-retriever.serviceAccountName" -}}
{{- if .Values.service.serviceAccount.create -}}
{{- default (include "nemo-retriever.fullname" .) .Values.service.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.service.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
=============================================================================
Label helpers
=============================================================================
*/}}

{{/*
nemo-retriever.labels
  Common labels applied to every object in the chart.
*/}}
{{- define "nemo-retriever.labels" -}}
helm.sh/chart: {{ include "nemo-retriever.chart" . }}
{{ include "nemo-retriever.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/part-of: nemo-retriever
{{- end -}}

{{/*
nemo-retriever.selectorLabels
  Selector labels for the service Deployment.  Stable across upgrades.
*/}}
{{- define "nemo-retriever.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nemo-retriever.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/component: service
{{- end -}}

{{/*
=============================================================================
PVC + Secret name helpers
=============================================================================
*/}}

{{- define "nemo-retriever.pvcName" -}}
{{- if .Values.persistence.existingClaim -}}
{{- .Values.persistence.existingClaim -}}
{{- else -}}
{{- printf "%s-data" (include "nemo-retriever.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.retrieverResultsPvcName" -}}
{{- if .Values.retrieverResults.existingClaim -}}
{{- .Values.retrieverResults.existingClaim -}}
{{- else -}}
{{- printf "%s-retriever-results" (include "nemo-retriever.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.nimApiKeySecretName" -}}
{{- if .Values.nimApiKey.existingSecret -}}
{{- .Values.nimApiKey.existingSecret -}}
{{- else -}}
{{- printf "%s-nim-api-key" (include "nemo-retriever.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.ngcApiKeySecretName" -}}
{{- if .Values.nims.ngcApiKey.existingSecret -}}
{{- .Values.nims.ngcApiKey.existingSecret -}}
{{- else -}}
{{- printf "%s-ngc-api-key" (include "nemo-retriever.fullname" .) -}}
{{- end -}}
{{- end -}}

{{- define "nemo-retriever.configMapName" -}}
{{- printf "%s-config" (include "nemo-retriever.fullname" .) -}}
{{- end -}}

{{/*
=============================================================================
Pull secret helpers
=============================================================================

Combine the user-supplied list of imagePullSecrets with the chart-managed
docker-config Secret (when imagePullSecret.create=true) and emit them in the
form expected by a Pod spec.
*/}}
{{- define "nemo-retriever.imagePullSecrets" -}}
{{- $secrets := list -}}
{{- if .Values.imagePullSecret.create -}}
{{- $secrets = append $secrets (dict "name" .Values.imagePullSecret.name) -}}
{{- end -}}
{{- range .Values.imagePullSecrets -}}
{{- $secrets = append $secrets . -}}
{{- end -}}
{{- if $secrets -}}
imagePullSecrets:
{{- range $secrets }}
  - name: {{ .name }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.nimImagePullSecrets
  Same as above but additionally folds in nims.imagePullSecrets so the NIM
  Deployments can opt into a separate registry.
*/}}
{{- define "nemo-retriever.nimImagePullSecrets" -}}
{{- $secrets := list -}}
{{- if .Values.imagePullSecret.create -}}
{{- $secrets = append $secrets (dict "name" .Values.imagePullSecret.name) -}}
{{- end -}}
{{- range .Values.imagePullSecrets -}}
{{- $secrets = append $secrets . -}}
{{- end -}}
{{- range .Values.nims.imagePullSecrets -}}
{{- $secrets = append $secrets . -}}
{{- end -}}
{{- if $secrets -}}
imagePullSecrets:
{{- range $secrets }}
  - name: {{ .name }}
{{- end }}
{{- end -}}
{{- end -}}

{{/*
=============================================================================
Split-topology helpers (gateway / realtime / batch)
=============================================================================
*/}}

{{/*
nemo-retriever.role.fullname
  Resource name for a topology role, e.g. <fullname>-gateway.
  Usage: {{ include "nemo-retriever.role.fullname" (dict "context" $ "role" "gateway") }}
*/}}
{{- define "nemo-retriever.role.fullname" -}}
{{- printf "%s-%s" (include "nemo-retriever.fullname" .context) .role | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.role.selectorLabels
  Stable selector labels for a topology-role Deployment / Service.
*/}}
{{- define "nemo-retriever.role.selectorLabels" -}}
app.kubernetes.io/name: {{ include "nemo-retriever.name" .context }}
app.kubernetes.io/instance: {{ .context.Release.Name }}
app.kubernetes.io/component: {{ .role }}
{{- end -}}

{{/*
nemo-retriever.role.labels
  Full labels for a topology-role resource.
*/}}
{{- define "nemo-retriever.role.labels" -}}
helm.sh/chart: {{ include "nemo-retriever.chart" .context }}
{{ include "nemo-retriever.role.selectorLabels" . }}
{{- if .context.Chart.AppVersion }}
app.kubernetes.io/version: {{ .context.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .context.Release.Service }}
app.kubernetes.io/part-of: nemo-retriever
{{- end -}}

{{/*
nemo-retriever.role.configMapName
  ConfigMap name for a topology role.
*/}}
{{- define "nemo-retriever.role.configMapName" -}}
{{- printf "%s-config" (include "nemo-retriever.role.fullname" .) -}}
{{- end -}}

{{/*
=============================================================================
NIM helpers
=============================================================================
*/}}

{{/*
nemo-retriever.nim.fullname
  Resource name for one NIM (e.g. <fullname>-nim-page-elements).
  Usage: {{ include "nemo-retriever.nim.fullname" (dict "context" $ "shortName" "page-elements") }}
*/}}
{{- define "nemo-retriever.nim.fullname" -}}
{{- $ctx := .context -}}
{{- $short := .shortName -}}
{{- printf "%s-nim-%s" (include "nemo-retriever.fullname" $ctx) $short | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
nemo-retriever.nim.selectorLabels
  Selector labels for one NIM Deployment.
*/}}
{{- define "nemo-retriever.nim.selectorLabels" -}}
{{- $ctx := .context -}}
{{- $short := .shortName -}}
app.kubernetes.io/name: {{ include "nemo-retriever.name" $ctx }}
app.kubernetes.io/instance: {{ $ctx.Release.Name }}
app.kubernetes.io/component: nim
nemo-retriever.nvidia.com/nim: {{ $short }}
{{- end -}}

{{/*
nemo-retriever.nim.labels
  Full labels for one NIM (selector labels + chart metadata).
*/}}
{{- define "nemo-retriever.nim.labels" -}}
helm.sh/chart: {{ include "nemo-retriever.chart" .context }}
{{ include "nemo-retriever.nim.selectorLabels" . }}
{{- if .context.Chart.AppVersion }}
app.kubernetes.io/version: {{ .context.Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .context.Release.Service }}
app.kubernetes.io/part-of: nemo-retriever
{{- end -}}

{{/*
nemo-retriever.nim.merged
  Returns the per-NIM values block merged on top of nims.defaults.
  Usage: {{ $cfg := include "nemo-retriever.nim.merged" (dict "context" $ "key" "pageElements") | fromYaml }}
*/}}
{{- define "nemo-retriever.nim.merged" -}}
{{- $ctx := .context -}}
{{- $key := .key -}}
{{- $defaults := deepCopy $ctx.Values.nims.defaults -}}
{{- $override := index $ctx.Values.nims $key -}}
{{- $merged := mergeOverwrite $defaults (deepCopy $override) -}}
{{- toYaml $merged -}}
{{- end -}}

{{/*
nemo-retriever.nim.url
  In-cluster invocation URL(s) for one NIM. When replicas > 1, returns
  comma-separated per-pod URLs so the retriever service can pin workers
  to individual NIM replicas (application-level load spreading). With a
  headless Service + StatefulSet the pod DNS is predictable:
    <name>-0.<name>.<namespace>.svc.cluster.local

  Returns "" when the NIM is disabled so callers can fall back to
  externally configured URLs.
*/}}
{{- define "nemo-retriever.nim.url" -}}
{{- $ctx := .context -}}
{{- $short := .shortName -}}
{{- $key := .key -}}
{{- $cfg := index $ctx.Values.nims $key -}}
{{- if and $ctx.Values.nims.enabled $cfg.enabled -}}
{{- $name := include "nemo-retriever.nim.fullname" (dict "context" $ctx "shortName" $short) -}}
{{- $merged := include "nemo-retriever.nim.merged" (dict "context" $ctx "key" $key) | fromYaml -}}
{{- $port := int $merged.port -}}
{{- $path := $cfg.invokePath -}}
{{- $replicas := int $merged.replicas -}}
{{- $urls := list -}}
{{- range $i := until $replicas -}}
{{- $urls = append $urls (printf "http://%s-%d.%s.%s.svc.cluster.local:%d%s" $name $i $name $ctx.Release.Namespace $port $path) -}}
{{- end -}}
{{- join "," $urls -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.nim.endpointURL
  Resolves the URL the service should call for a given NIM endpoint.

  Resolution order:
    1. An explicit override in .Values.serviceConfig.nimEndpoints.<key>.
    2. The auto-generated in-cluster URL when nims.enabled and the
       individual NIM is enabled.
    3. Empty string (the service treats this as "no endpoint configured").

  Usage: {{ include "nemo-retriever.nim.endpointURL" (dict "context" $ "key" "pageElements" "shortName" "page-elements" "configKey" "pageElementsInvokeUrl") }}
*/}}
{{- define "nemo-retriever.nim.endpointURL" -}}
{{- $ctx := .context -}}
{{- $explicit := index $ctx.Values.serviceConfig.nimEndpoints .configKey -}}
{{- if $explicit -}}
{{- $explicit -}}
{{- else -}}
{{- include "nemo-retriever.nim.url" (dict "context" $ctx "shortName" .shortName "key" .key) -}}
{{- end -}}
{{- end -}}

{{/*
=============================================================================
Runtime selector (python | rust)
=============================================================================
*/}}

{{/*
nemo-retriever.runtime
  Resolved runtime name. Falls back to "python" when unset or invalid.
*/}}
{{- define "nemo-retriever.runtime" -}}
{{- $r := default "python" .Values.runtime -}}
{{- if and (ne $r "python") (ne $r "rust") -}}
{{- $r = "python" -}}
{{- end -}}
{{- $r -}}
{{- end -}}

{{/*
nemo-retriever.image.repository
  Resolves to runtimeImages.<runtime>.repository, with a back-compat fallback
  to the legacy service.image.repository for python.
*/}}
{{- define "nemo-retriever.image.repository" -}}
{{- $rt := include "nemo-retriever.runtime" . -}}
{{- $img := index .Values.runtimeImages $rt -}}
{{- if and (eq $rt "python") (and (hasKey .Values "service") (hasKey .Values.service "image") .Values.service.image.repository) -}}
{{- .Values.service.image.repository -}}
{{- else -}}
{{- $img.repository -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.image.tag
  Resolves to runtimeImages.<runtime>.tag, with a back-compat fallback to
  the legacy service.image.tag for python.
*/}}
{{- define "nemo-retriever.image.tag" -}}
{{- $rt := include "nemo-retriever.runtime" . -}}
{{- $img := index .Values.runtimeImages $rt -}}
{{- if and (eq $rt "python") (and (hasKey .Values "service") (hasKey .Values.service "image") .Values.service.image.tag) -}}
{{- .Values.service.image.tag -}}
{{- else -}}
{{- $img.tag -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.image.pullPolicy
  Resolves runtimeImages.<runtime>.pullPolicy with the same fallback rule.
*/}}
{{- define "nemo-retriever.image.pullPolicy" -}}
{{- $rt := include "nemo-retriever.runtime" . -}}
{{- $img := index .Values.runtimeImages $rt -}}
{{- if and (eq $rt "python") (and (hasKey .Values "service") (hasKey .Values.service "image") .Values.service.image.pullPolicy) -}}
{{- .Values.service.image.pullPolicy -}}
{{- else -}}
{{- $img.pullPolicy -}}
{{- end -}}
{{- end -}}

{{/*
nemo-retriever.container.args
  Returns the container `args` slice for the chosen runtime as a YAML
  block. Both runtimes accept `--config <path>`. The Rust binary is
  `retriever-rs` and the Python binary is `retriever`.
*/}}
{{- define "nemo-retriever.container.args" -}}
{{- $rt := include "nemo-retriever.runtime" . -}}
{{- if eq $rt "rust" }}
{{- /*
The Rust image's ENTRYPOINT is the `retriever-rs` binary itself, so its
container `args` are the binary's argv (no leading program name).
*/ -}}
- service
- start
- --config
- /etc/nemo-retriever/retriever-service.yaml
{{- else }}
{{- /*
The Python image's ENTRYPOINT runs a shell, so the program name is
required as the first arg.
*/ -}}
- retriever
- service
- start
- --config
- /etc/nemo-retriever/retriever-service.yaml
{{- end }}
{{- end -}}
