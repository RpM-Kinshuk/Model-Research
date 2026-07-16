from alpha_cluster_dashboard.data import canonicalize_module_name


def test_canonicalize_common_gpt_style_modules():
    assert canonicalize_module_name("attn.c_attn_q") == "attn_q"
    assert canonicalize_module_name("attn.c_attn_k") == "attn_k"
    assert canonicalize_module_name("attn.c_attn_v") == "attn_v"
    assert canonicalize_module_name("attn.c_proj") == "attn_o"
    assert canonicalize_module_name("mlp.c_fc") == "mlp_up"
    assert canonicalize_module_name("mlp.c_proj") == "mlp_down"


def test_canonicalize_common_gpt_neox_modules():
    assert canonicalize_module_name("attention.query_key_value_q") == "attn_q"
    assert canonicalize_module_name("attention.query_key_value_k") == "attn_k"
    assert canonicalize_module_name("attention.query_key_value_v") == "attn_v"
    assert canonicalize_module_name("attention.dense") == "attn_o"
    assert canonicalize_module_name("mlp.dense_h_to_4h") == "mlp_up"
    assert canonicalize_module_name("mlp.dense_4h_to_h") == "mlp_down"


def test_canonicalize_common_opt_modules():
    assert canonicalize_module_name("fc1") == "mlp_up"
    assert canonicalize_module_name("fc2") == "mlp_down"
