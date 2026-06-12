from __future__ import annotations

import streamlit as st
import utils.model_lab_workflows as mlw


def render_feature_checklist(context):
    config = context['config']
    editable = context.get('is_new_model') or context['status'] in mlw.EDITABLE_STATUSES
    model_key = str(context.get('model_id') or 'new_model').replace('-', '_')
    feature_config = config.get('features') or {}
    current = set(feature_config.get('feature_columns') or [])
    available = mlw._available_feature_columns(context)
    bundle_map = mlw._bundle_map(available)
    saved = list(feature_config.get('selected_bundles') or [])
    if not saved:
        saved = mlw._infer_selected_bundles(config, available)
    saved_set = set(saved)

    st.markdown('#### Feature Selection')
    selected = st.multiselect(
        'Selected Bundles',
        list(bundle_map.keys()),
        default=[b for b in saved if b in bundle_map],
        format_func=lambda b: f"{mlw.BUNDLE_LABELS.get(b, b)} ({len(bundle_map.get(b, []))})",
        disabled=not editable,
        key=f'mlab_selected_bundles_{model_key}',
    )

    included = []
    removed = []
    universe = []
    for bundle in selected:
        features = list(bundle_map.get(bundle, []))
        universe.extend(features)
        with st.expander(f"{mlw.BUNDLE_LABELS.get(bundle, bundle)} ({len(features)} features)", expanded=True):
            cols = st.columns(3)
            for i, feature in enumerate(features):
                default = feature in current if bundle in saved_set else True
                key = f"mlab_feature_{model_key}_{bundle}_{feature}".replace(' ', '_').replace('-', '_')
                with cols[i % 3]:
                    value = st.checkbox(feature, value=default, disabled=not editable, key=key)
                if value:
                    included.append(feature)
                else:
                    removed.append(feature)

    resolved = list(dict.fromkeys(included))
    removed = list(dict.fromkeys(removed))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Bundles', len(selected))
    c2.metric('Available', len(list(dict.fromkeys(universe))))
    c3.metric('Included', len(resolved))
    c4.metric('Unchecked', len(removed))
    return {'selected_bundles': selected, 'include_features': [], 'exclude_features': removed, 'resolved_features': resolved}
