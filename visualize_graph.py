from graph import app

# Generate Mermaid markdown
mermaid_md = app.get_graph().draw_mermaid()
print("--- GRAPH MERMAID DIAGRAM ---")
print(mermaid_md)
print("-----------------------------")

# To save as PNG (requires pygraphviz or similar, usually easier to just use Mermaid markdown)
# with open("graph.png", "wb") as f:
#     f.write(app.get_graph().draw_mermaid_png())
