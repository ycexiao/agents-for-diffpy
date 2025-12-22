from FitDAG import FitDAG

dag = FitDAG()
dag.from_workflow("debug_dag.pkl")
dag.render("debug_dag.html")
