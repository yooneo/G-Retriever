def load_dataset(dataset_name):
    if dataset_name == 'expla_graphs':
        from src.dataset.expla_graphs import ExplaGraphsDataset
        return ExplaGraphsDataset
    elif dataset_name == 'scene_graphs':
        from src.dataset.scene_graphs import SceneGraphsDataset
        return SceneGraphsDataset
    elif dataset_name == 'scene_graphs_baseline':
        from src.dataset.scene_graphs_baseline import SceneGraphsBaselineDataset
        return SceneGraphsBaselineDataset
    elif dataset_name == 'webqsp':
        from src.dataset.webqsp import WebQSPDataset
        return WebQSPDataset
    elif dataset_name == 'webqsp_baseline':
        from src.dataset.webqsp_baseline import WebQSPBaselineDataset
        return WebQSPBaselineDataset
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
