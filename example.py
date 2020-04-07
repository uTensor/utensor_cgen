from ortools.sat.python import cp_model

model = cp_model.CpModel()
model.NewIntVar      # declare a variable of type int
model.NewBoolVar     # declare a variable of type bool
model.NewIntervalVar # declare an inclusive interval [a, b]
model.AddNoOverlap   # add constraint for non overlapping intervals
# and more...

x_start = model.NewIntVar(0, 4, 'x_start')
x_end = model.NewIntVar(0, 4, 'x_end')
x_interval = model.NewIntervalVar(x_start, 1, x_end, 'x_interval')
y_start = model.NewIntVar(0, 4, 'y_start')
y_end = model.NewIntVar(0, 4, 'y_end')
y_interval = model.NewIntervalVar(y_start, 1, y_end, 'x_interval')
model.AddNoOverlap([x_interval, y_interval])
total_span = model.NewIntVar(0, 4, 'total_span')
model.AddMaxEquality(total_span, [x_end, y_end])
model.Minimize(total_span)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if solver.StatusName(status) == 'OPTIMAL':
    print('x interval:', solver.Value(x_start), solver.Value(x_end))
    print('y interval:', solver.Value(y_start), solver.Value(y_end))
    print('total span:', solver.Value(total_span))
else:
    print('no optimal solution')
