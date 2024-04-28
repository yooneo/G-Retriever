load_dataset = {
    'expla_graphs': lambda: __import__('src.dataset.expla_graphs').ExplaGraphsDataset,
    'scene_graphs': lambda: __import__('src.dataset.scene_graphs').SceneGraphsDataset,
    'scene_graphs_baseline': lambda: __import__('src.dataset.scene_graphs_baseline').SceneGraphsBaselineDataset,
    'webqsp': lambda: __import__('src.dataset.webqsp').WebQSPDataset,
    'webqsp_baseline': lambda: __import__('src.dataset.webqsp_baseline').WebQSPBaselineDataset,
}

