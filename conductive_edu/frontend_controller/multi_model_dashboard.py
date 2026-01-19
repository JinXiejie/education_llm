
import gradio as gr
import ollama


class OllamaMultiModelDashboard:
    """Ollama 多模型控制面板"""

    def __init__(self):
        self.models_info = {}
        self.update_models()

    def update_models(self):
        """更新模型信息"""
        try:
            models = ollama.list()
            for model in models['models']:
                self.models_info[model['name']] = {
                    'size': model.get('size', 0),
                    'modified': model.get('modified_at', '')
                }
        except:
            pass

    def pull_model(self, model_name: str):
        """拉取模型"""
        try:
            response = ollama.pull(model_name, stream=True)
            progress = ""
            for line in response:
                if 'status' in line:
                    progress += f"{line['status']}\n"
                    yield progress
            yield f"✅ 模型 {model_name} 下载完成！"
        except Exception as e:
            yield f"❌ 下载失败: {str(e)}"

    def delete_model(self, model_name: str):
        """删除模型"""
        try:
            ollama.delete(model_name)
            self.update_models()
            return f"✅ 已删除模型: {model_name}"
        except Exception as e:
            return f"❌ 删除失败: {str(e)}"

    def create_dashboard(self):
        """创建控制面板"""

        with gr.Blocks(title="Ollama 模型管理面板") as demo:
            gr.Markdown("""
            # 🛠️ Ollama 模型管理面板
            管理本地的大语言模型
            """)

            with gr.Tabs():
                with gr.TabItem("📊 模型列表"):
                    gr.Markdown("### 本地已安装模型")
                    models_table = gr.Dataframe(
                        headers=["模型名称", "大小", "修改时间"],
                        value=self.get_models_table_data(),
                        interactive=False
                    )

                    refresh_btn = gr.Button("🔄 刷新列表")

                    refresh_btn.click(
                        self.update_models_and_table,
                        None,
                        [models_table]
                    )

                with gr.TabItem("⬇️ 下载模型"):
                    gr.Markdown("### 下载新模型")

                    available_models = [
                        "llama2", "llama2:13b", "llama2:70b",
                        "mistral", "codellama", "neural-chat",
                        "starling-lm", "orca-mini"
                    ]

                    with gr.Row():
                        model_to_pull = gr.Dropdown(
                            choices=available_models,
                            label="选择要下载的模型"
                        )
                        pull_btn = gr.Button("开始下载", variant="primary")

                    pull_output = gr.Textbox(
                        label="下载进度",
                        lines=10,
                        interactive=False
                    )

                    pull_btn.click(
                        self.pull_model,
                        [model_to_pull],
                        [pull_output]
                    )

                with gr.TabItem("🗑️ 删除模型"):
                    gr.Markdown("### 删除本地模型")

                    with gr.Row():
                        model_to_delete = gr.Dropdown(
                            choices=list(self.models_info.keys()),
                            label="选择要删除的模型"
                        )
                        delete_btn = gr.Button("删除模型", variant="stop")

                    delete_output = gr.Textbox(
                        label="操作结果",
                        interactive=False
                    )

                    delete_btn.click(
                        self.delete_model,
                        [model_to_delete],
                        [delete_output]
                    ).then(
                        self.update_models_and_table,
                        None,
                        [models_table]
                    )

        return demo

    def get_models_table_data(self):
        """获取模型表格数据"""
        data = []
        for name, info in self.models_info.items():
            size_mb = info.get('size', 0) / 1024 / 1024
            data.append([
                name,
                f"{size_mb:.1f} MB",
                info.get('modified', '')
            ])
        return data

    def update_models_and_table(self):
        """更新模型信息并刷新表格"""
        self.update_models()
        return self.get_models_table_data()
