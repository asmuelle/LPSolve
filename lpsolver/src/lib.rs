// Import wasm-bindgen prelude
use wasm_bindgen::prelude::*;

// Utility for logging from wasm to the JS console
#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Represents the type of optimization for the objective function.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectiveType {
    Maximize,
    Minimize,
}

/// Represents the objective function of a linear programming problem.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct ObjectiveFunction {
    /// Coefficients of the variables in the objective function.
    #[wasm_bindgen(skip)]
    pub coefficients: Vec<f64>,
    /// The type of optimization (Maximize or Minimize).
    #[wasm_bindgen(skip)]
    pub objective_type: ObjectiveType,
}

#[wasm_bindgen]
impl ObjectiveFunction {
    /// Creates a new ObjectiveFunction.
    #[wasm_bindgen(constructor)]
    pub fn new(coefficients: Vec<f64>, objective_type: ObjectiveType) -> Self {
        ObjectiveFunction {
            coefficients,
            objective_type,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn objective_type(&self) -> ObjectiveType {
        self.objective_type
    }
}

/// Represents the type of a constraint.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintType {
    LessThanOrEqual,
    EqualTo,
    GreaterThanOrEqual,
}

/// Represents a single constraint in a linear programming problem.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct Constraint {
    #[wasm_bindgen(skip)]
    pub coefficients: Vec<f64>,
    #[wasm_bindgen(skip)]
    pub constraint_type: ConstraintType,
    #[wasm_bindgen(skip)]
    pub rhs: f64,
}

#[wasm_bindgen]
impl Constraint {
    /// Creates a new Constraint.
    #[wasm_bindgen(constructor)]
    pub fn new(coefficients: Vec<f64>, constraint_type: ConstraintType, rhs: f64) -> Self {
        Constraint {
            coefficients,
            constraint_type,
            rhs,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn coefficients(&self) -> Vec<f64> {
        self.coefficients.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn constraint_type(&self) -> ConstraintType {
        self.constraint_type
    }

    #[wasm_bindgen(getter)]
    pub fn rhs(&self) -> f64 {
        self.rhs
    }
}

/// Represents a linear programming (LP) problem.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct LPProblem {
    #[wasm_bindgen(skip)]
    pub objective: ObjectiveFunction,
    #[wasm_bindgen(skip)]
    pub constraints: Vec<Constraint>,
}

#[wasm_bindgen]
impl LPProblem {
    /// Creates a new LPProblem.
    #[wasm_bindgen(constructor)]
    pub fn new(objective: ObjectiveFunction, constraints: Vec<Constraint>) -> Self {
        LPProblem {
            objective,
            constraints,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn objective(&self) -> ObjectiveFunction {
        self.objective.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn constraints(&self) -> Vec<Constraint> {
        self.constraints.clone()
    }
}

/// Represents the result status of the Simplex algorithm.
#[wasm_bindgen]
#[derive(Clone, Copy, Debug, PartialEq, Eq)] // Added Copy
pub enum LPSolutionStatus {
    Optimal,
    Unbounded,
    Infeasible,
    MaxIterationsReached, // Added for clarity
}

/// Represents the solution of an LP problem.
#[wasm_bindgen]
#[derive(Clone, Debug)]
pub struct LPSolution {
    pub status: LPSolutionStatus,
    pub objective_value: Option<f64>,
    pub variable_values: Option<Vec<f64>>,
}

#[wasm_bindgen]
impl LPSolution {
    #[wasm_bindgen(constructor)] // Allow JS to create this, though typically Rust creates it
    pub fn new(status: LPSolutionStatus, objective_value: Option<f64>, variable_values: Option<Vec<f64>>) -> Self {
        LPSolution { status, objective_value, variable_values }
    }
    
    // Internal constructors
    pub(crate) fn optimal(objective_value: f64, variable_values: Vec<f64>) -> Self {
        LPSolution {
            status: LPSolutionStatus::Optimal,
            objective_value: Some(objective_value),
            variable_values: Some(variable_values),
        }
    }

    pub(crate) fn unbounded() -> Self {
        LPSolution {
            status: LPSolutionStatus::Unbounded,
            objective_value: None,
            variable_values: None,
        }
    }

    pub(crate) fn infeasible() -> Self {
        LPSolution {
            status: LPSolutionStatus::Infeasible,
            objective_value: None,
            variable_values: None,
        }
    }
    
    pub(crate) fn max_iterations_reached() -> Self {
        LPSolution {
            status: LPSolutionStatus::MaxIterationsReached,
            objective_value: None,
            variable_values: None,
        }
    }


    #[wasm_bindgen(getter)]
    pub fn status(&self) -> LPSolutionStatus {
        self.status
    }

    #[wasm_bindgen(getter)]
    pub fn objective_value(&self) -> Option<f64> {
        self.objective_value
    }

    #[wasm_bindgen(getter)]
    pub fn variable_values(&self) -> Option<Vec<f64>> {
        self.variable_values.clone()
    }
}

/// Implements the Simplex algorithm.
#[wasm_bindgen]
pub struct SimplexSolver {
    tableau: Vec<Vec<f64>>,
    // Total variables in tableau (original + slack/artificial)
    num_tableau_variables: usize,
    num_constraints: usize,
    // Indices of basic variables for each row
    basic_variables: Vec<usize>,
    // Store original problem characteristics for solution interpretation
    original_objective_type: ObjectiveType,
    num_original_variables: usize,
}

#[wasm_bindgen]
impl SimplexSolver {
    #[wasm_bindgen(constructor)]
    pub fn new(problem: &LPProblem) -> Result<SimplexSolver, JsValue> {
        let num_original_variables = problem.objective.coefficients.len();
        let num_constraints = problem.constraints.len();

        // Standard form requires all variables to be non-negative. This is assumed.
        // Standard form requires RHS to be non-negative.
        for constraint in &problem.constraints {
            if constraint.rhs < 0.0 {
                return Err(JsValue::from_str("Negative RHS values are not supported. Problem must be in standard form or requires transformation (e.g., multiply constraint by -1 and flip inequality)."));
            }
        }
        
        // This basic implementation only handles LessThanOrEqual constraints directly
        // by adding slack variables. Equality and GreaterThanOrEqual constraints
        // would require artificial variables and a two-phase or Big-M method.
        for constraint in &problem.constraints {
            if constraint.constraint_type != ConstraintType::LessThanOrEqual {
                return Err(JsValue::from_str("Only LessThanOrEqual constraints are directly supported in this basic Simplex version. Other types require methods like Two-Phase Simplex or Big-M."));
            }
        }

        // Number of variables in the tableau (original + slack for each constraint)
        let num_tableau_variables = num_original_variables + num_constraints;
        
        let mut tableau = vec![vec![0.0; num_tableau_variables + 1]; num_constraints + 1];
        let mut basic_variables = vec![0; num_constraints]; // Stores column index of basic var for the row

        // Initialize objective function row
        // If ObjectiveType is Minimize, we convert to Maximize (-Objective).
        // The tableau's last row stores coefficients for this maximization problem.
        // Standard simplex for maximization expects -c_j in the objective row if starting from Z - cX = 0.
        // Or c_j if starting from -Z + cX = 0. Let's use the common -c_j.
        let multiplier = match problem.objective.objective_type {
            ObjectiveType::Maximize => 1.0,
            ObjectiveType::Minimize => -1.0, // We will maximize (-original_objective)
        };

        for (j, &coeff) in problem.objective.coefficients.iter().enumerate() {
            tableau[num_constraints][j] = -coeff * multiplier;
        }
        // tableau[num_constraints][num_tableau_variables] is the initial objective value (0)

        // Initialize constraint rows
        for i in 0..num_constraints {
            let constraint = &problem.constraints[i];
            if constraint.coefficients.len() != num_original_variables {
                return Err(JsValue::from_str("Constraint coefficient count mismatch with number of objective variables."));
            }

            for (j, &coeff) in constraint.coefficients.iter().enumerate() {
                tableau[i][j] = coeff;
            }

            // Add slack variable for this LessThanOrEqual constraint
            tableau[i][num_original_variables + i] = 1.0;
            basic_variables[i] = num_original_variables + i; // The slack variable for row i is basic
            
            tableau[i][num_tableau_variables] = constraint.rhs;
        }
        
        Ok(SimplexSolver {
            tableau,
            num_tableau_variables,
            num_constraints,
            basic_variables,
            original_objective_type: problem.objective.objective_type,
            num_original_variables,
        })
    }

    /// Solves the LP problem using the Simplex algorithm.
    pub fn solve(&mut self) -> LPSolution {
        let max_iterations = 1000; // Safeguard against cycling

        for _iter in 0..max_iterations {
            // Find pivot column (entering variable)
            // Most negative coefficient in the objective row (last row of tableau)
            let mut pivot_col_idx = usize::MAX;
            let mut min_obj_coeff = -1e-9; // Allow for small negative due to precision

            // Iterate only over non-basic variables in objective row (original + slack)
            for j in 0..self.num_tableau_variables {
                if self.tableau[self.num_constraints][j] < min_obj_coeff {
                    min_obj_coeff = self.tableau[self.num_constraints][j];
                    pivot_col_idx = j;
                }
            }

            // If no negative coefficient, solution is optimal
            if pivot_col_idx == usize::MAX {
                return self.extract_solution();
            }

            // Find pivot row (leaving variable) using minimum ratio test
            let mut pivot_row_idx = usize::MAX;
            let mut min_ratio = f64::MAX;

            for i in 0..self.num_constraints {
                let pivot_col_val = self.tableau[i][pivot_col_idx];
                if pivot_col_val > 1e-9 { // Denominator must be positive
                    let rhs_val = self.tableau[i][self.num_tableau_variables];
                    let ratio = rhs_val / pivot_col_val;
                    if ratio < min_ratio {
                        min_ratio = ratio;
                        pivot_row_idx = i;
                    }
                }
            }

            // If no suitable pivot row, problem is unbounded
            if pivot_row_idx == usize::MAX {
                return LPSolution::unbounded();
            }

            // Perform pivot operation
            self.pivot(pivot_row_idx, pivot_col_idx);
            // Update basic variable for the pivot row
            self.basic_variables[pivot_row_idx] = pivot_col_idx;
        }

        LPSolution::max_iterations_reached()
    }

    /// Performs one pivot operation.
    fn pivot(&mut self, pivot_row: usize, pivot_col: usize) {
        let pivot_element = self.tableau[pivot_row][pivot_col];
        
        // Normalize pivot row
        for j in 0..=self.num_tableau_variables { // Iterate up to and including RHS column
            self.tableau[pivot_row][j] /= pivot_element;
        }

        // Eliminate other rows (including objective row)
        for i in 0..=self.num_constraints { // Iterate all rows
            if i != pivot_row {
                let factor = self.tableau[i][pivot_col];
                for j in 0..=self.num_tableau_variables { // Iterate all columns
                    self.tableau[i][j] -= factor * self.tableau[pivot_row][j];
                }
            }
        }
    }
    
    /// Extracts solution from the tableau.
    fn extract_solution(&self) -> LPSolution {
        let mut variable_values = vec![0.0; self.num_original_variables];
        
        for i in 0..self.num_constraints {
            let basic_var_idx = self.basic_variables[i]; // Column index of basic variable in this row
            // If the basic variable is one of the original decision variables
            if basic_var_idx < self.num_original_variables {
                 variable_values[basic_var_idx] = self.tableau[i][self.num_tableau_variables]; // RHS value
            }
        }
        
        let mut objective_value_from_tableau = self.tableau[self.num_constraints][self.num_tableau_variables];

        // If original problem was minimization, the tableau maximized (-objective).
        // So, the optimal value for the original minimization problem is -value_from_tableau.
        if self.original_objective_type == ObjectiveType::Minimize {
            objective_value_from_tableau = -objective_value_from_tableau;
        }

        LPSolution::optimal(objective_value_from_tableau, variable_values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to create a simple LP problem for testing Simplex
    fn create_simple_max_problem() -> LPProblem {
        // Maximize P = 3x + 2y
        // Subject to:
        // x + y <= 4
        // 2x + y <= 5
        // x, y >= 0
        let objective = ObjectiveFunction::new(vec![3.0, 2.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0, 1.0], ConstraintType::LessThanOrEqual, 4.0),
            Constraint::new(vec![2.0, 1.0], ConstraintType::LessThanOrEqual, 5.0),
        ];
        LPProblem::new(objective, constraints)
    }
    
    fn create_simple_min_problem() -> LPProblem {
        // Minimize Z = x - 3y + 2z  (Solution from online solver: Z = -11 at x=4, y=5, z=0, if constraints were different)
        // Actual example: Min Z = 2x - y s.t. x+y <= 3, x-y <= 1, x,y >=0. Sol: Z = -3 (x=0, y=3)
        // Let's use: Min Z = x - 2y
        // s.t.  x + y <= 4   (s1)
        //      2x + y <= 5   (s2)
        //      x,y >= 0
        // Max -Z = -x + 2y
        // Optimal for Max P = 3x+2y is (x=1, y=3, P=9)
        // Let's use problem: Min 2x + 3y s.t. x >= 1, y >= 1, x+y <=3 (This needs >= constraints, not supported yet)
        
        // Let's use a known simple minimization problem that fits <= constraints:
        // Minimize C = -3x - 5y
        // s.t. x <= 4  (x + s1 = 4)
        //      y <= 6  (y + s2 = 6)
        //      3x + 2y <= 18 (3x + 2y + s3 = 18)
        //      x,y >=0
        // Solution: x=2, y=6, C = -3*2 - 5*6 = -6 -30 = -36. (from online calc)
        // Max -C = 3x + 5y
        let objective = ObjectiveFunction::new(vec![-3.0, -5.0], ObjectiveType::Minimize); // Min -3x-5y
        let constraints = vec![
            Constraint::new(vec![1.0, 0.0], ConstraintType::LessThanOrEqual, 4.0), // x <= 4
            Constraint::new(vec![0.0, 1.0], ConstraintType::LessThanOrEqual, 6.0), // y <= 6
            Constraint::new(vec![3.0, 2.0], ConstraintType::LessThanOrEqual, 18.0), // 3x+2y <= 18
        ];
        LPProblem::new(objective, constraints)
    }


    #[test]
    fn test_simplex_solver_new_max_problem() {
        let problem = create_simple_max_problem(); // Max 3x+2y, x+y<=4, 2x+y<=5
        let solver_result = SimplexSolver::new(&problem);
        assert!(solver_result.is_ok());
        let solver = solver_result.unwrap();

        assert_eq!(solver.num_original_variables, 2);
        assert_eq!(solver.num_constraints, 2);
        assert_eq!(solver.num_tableau_variables, 4); // 2 original + 2 slack
        assert_eq!(solver.original_objective_type, ObjectiveType::Maximize);

        // Tableau: (num_constraints+1) x (num_tableau_variables+1) = 3x5
        // Obj row: -c_j for Max. So, -3, -2 for x,y. 0,0 for s1,s2. RHS 0.
        // [-3, -2, 0, 0 | 0]
        assert_eq!(solver.tableau[2], vec![-3.0, -2.0, 0.0, 0.0, 0.0]);
        // Constraints:
        // x + y + s1 = 4  => [1, 1, 1, 0 | 4]
        assert_eq!(solver.tableau[0], vec![1.0, 1.0, 1.0, 0.0, 4.0]);
        // 2x + y + s2 = 5 => [2, 1, 0, 1 | 5]
        assert_eq!(solver.tableau[1], vec![2.0, 1.0, 0.0, 1.0, 5.0]);
        
        assert_eq!(solver.basic_variables, vec![2, 3]); // s1 (idx 2), s2 (idx 3)
    }
    
    #[test]
    fn test_simplex_solver_new_min_problem() {
        let problem = create_simple_min_problem(); // Min -3x-5y => Max 3x+5y
        let solver_result = SimplexSolver::new(&problem);
        assert!(solver_result.is_ok());
        let solver = solver_result.unwrap();

        assert_eq!(solver.num_original_variables, 2);
        assert_eq!(solver.num_constraints, 3);
        assert_eq!(solver.num_tableau_variables, 5); // 2 original + 3 slack
        assert_eq!(solver.original_objective_type, ObjectiveType::Minimize);

        // Tableau for Max 3x+5y. Obj row: -c'_j => -3, -5.
        // [-3, -5, 0, 0, 0 | 0]
        assert_eq!(solver.tableau[3], vec![-3.0, -5.0, 0.0, 0.0, 0.0, 0.0]);
        // Constraints:
        // x + s1 = 4       => [1,0,1,0,0 | 4]
        assert_eq!(solver.tableau[0], vec![1.0, 0.0, 1.0, 0.0, 0.0, 4.0]);
        // y + s2 = 6       => [0,1,0,1,0 | 6]
        assert_eq!(solver.tableau[1], vec![0.0, 1.0, 0.0, 1.0, 0.0, 6.0]);
        // 3x + 2y + s3 = 18 => [3,2,0,0,1 | 18]
        assert_eq!(solver.tableau[2], vec![3.0, 2.0, 0.0, 0.0, 1.0, 18.0]);
        
        assert_eq!(solver.basic_variables, vec![2,3,4]); // s1,s2,s3 (indices 2,3,4)
    }


    #[test]
    fn test_simplex_solve_simple_max_problem() {
        let problem = create_simple_max_problem(); // Max 3x+2y, x+y<=4, 2x+y<=5
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        // Sol: x=1, y=3, P=9
        assert_eq!(solution.objective_value, Some(9.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 2);
            assert!((values[0] - 1.0).abs() < 1e-6, "x value was {}", values[0]); // x = 1
            assert!((values[1] - 3.0).abs() < 1e-6, "y value was {}", values[1]); // y = 3
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }
    
    #[test]
    fn test_simplex_solve_simple_min_problem() {
        let problem = create_simple_min_problem(); // Min -3x-5y (Max 3x+5y)
                                                  // s.t. x<=4, y<=6, 3x+2y<=18. Sol: x=2, y=6, C = -36
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert_eq!(solution.objective_value, Some(-36.0)); // Original Min value
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 2);
            assert!((values[0] - 2.0).abs() < 1e-6, "x value was {}", values[0]); // x = 2
            assert!((values[1] - 6.0).abs() < 1e-6, "y value was {}", values[1]); // y = 6
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }

    #[test]
    fn test_simplex_unbounded() {
        // Max P = x + y
        // s.t. x - y <= 1
        //     -x + y <= 2 (y - x <= 2)
        // x,y >=0
        let objective = ObjectiveFunction::new(vec![1.0, 1.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0, -1.0], ConstraintType::LessThanOrEqual, 1.0),
            Constraint::new(vec![-1.0, 1.0], ConstraintType::LessThanOrEqual, 2.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();
        assert_eq!(solution.status, LPSolutionStatus::Unbounded);
    }

    #[test]
    fn test_simplex_error_on_unsupported_constraint_type() {
         let objective = ObjectiveFunction::new(vec![1.0, 1.0], ObjectiveType::Maximize);
         let constraints = vec![
             Constraint::new(vec![1.0, 1.0], ConstraintType::GreaterThanOrEqual, 2.0), // Not supported
         ];
         let problem = LPProblem::new(objective, constraints);
         let solver_result = SimplexSolver::new(&problem);
         assert!(solver_result.is_err());
         if let Err(js_val) = solver_result {
            assert!(js_val.as_string().unwrap_or_default().contains("Only LessThanOrEqual constraints"));
         }
    }

    #[test]
    fn test_simplex_error_on_negative_rhs() {
         let objective = ObjectiveFunction::new(vec![1.0, 1.0], ObjectiveType::Maximize);
         let constraints = vec![
             Constraint::new(vec![1.0, 1.0], ConstraintType::LessThanOrEqual, -2.0), // Negative RHS
         ];
         let problem = LPProblem::new(objective, constraints);
         let solver_result = SimplexSolver::new(&problem);
         assert!(solver_result.is_err());
         if let Err(js_val) = solver_result {
            assert!(js_val.as_string().unwrap_or_default().contains("Negative RHS values are not supported"));
         }
    }

    #[test]
    fn test_optimal_solution_at_origin() {
        // Maximize P = -x - y  (or x + y for Minimize)
        // Subject to:
        // x <= 0 (implicitly, or add x >= 0, then solution is x=0, y=0)
        // y <= 0 (implicitly)
        // For this test, let's use: Max P = -x -y s.t. x + s1 = 0, y + s2 = 0 (not good constraints)
        // Better: Maximize P = -x - y subject to x >= 0, y >= 0. The solver assumes non-negativity.
        // If objective is -x -y, and x,y must be >=0, then optimal is x=0, y=0, P=0.
        // Constraints like x <= 1, y <= 1 would make it x=0,y=0, P=0.
        let objective = ObjectiveFunction::new(vec![-1.0, -1.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0, 0.0], ConstraintType::LessThanOrEqual, 10.0), // x <= 10
            Constraint::new(vec![0.0, 1.0], ConstraintType::LessThanOrEqual, 10.0), // y <= 10
        ]; // With x,y >= 0 implied, solution for Max -x-y is x=0, y=0, P=0.
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert_eq!(solution.objective_value, Some(0.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 2);
            assert!((values[0] - 0.0).abs() < 1e-6); // x = 0
            assert!((values[1] - 0.0).abs() < 1e-6); // y = 0
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }

    #[test]
    fn test_3_variable_problem() {
        // Max P = 5x1 + 4x2 + 3x3
        // s.t. 2x1 + 3x2 + x3 <= 5
        //      4x1 + x2 + 2x3 <= 11
        //      3x1 + 4x2 + 2x3 <= 8
        //      x1,x2,x3 >= 0
        // Solution: x1=2, x2=0, x3=1, P=13
        let objective = ObjectiveFunction::new(vec![5.0, 4.0, 3.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![2.0, 3.0, 1.0], ConstraintType::LessThanOrEqual, 5.0),
            Constraint::new(vec![4.0, 1.0, 2.0], ConstraintType::LessThanOrEqual, 11.0),
            Constraint::new(vec![3.0, 4.0, 2.0], ConstraintType::LessThanOrEqual, 8.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert_eq!(solution.objective_value, Some(13.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 3);
            assert!((values[0] - 2.0).abs() < 1e-6, "x1 value was {}", values[0]); // x1 = 2
            assert!((values[1] - 0.0).abs() < 1e-6, "x2 value was {}", values[1]); // x2 = 0
            assert!((values[2] - 1.0).abs() < 1e-6, "x3 value was {}", values[2]); // x3 = 1
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }
    
    #[test]
    fn test_redundant_constraint() {
        // Maximize P = 3x + 2y
        // Subject to:
        // x + y <= 4
        // 2x + y <= 5
        // x <= 10 (redundant)
        // x, y >= 0
        // Solution should be same as create_simple_max_problem: x=1, y=3, P=9
        let objective = ObjectiveFunction::new(vec![3.0, 2.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0, 1.0], ConstraintType::LessThanOrEqual, 4.0),
            Constraint::new(vec![2.0, 1.0], ConstraintType::LessThanOrEqual, 5.0),
            Constraint::new(vec![1.0, 0.0], ConstraintType::LessThanOrEqual, 10.0), // Redundant
        ];
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert_eq!(solution.objective_value, Some(9.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 2);
            assert!((values[0] - 1.0).abs() < 1e-6);
            assert!((values[1] - 3.0).abs() < 1e-6);
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }

    #[test]
    fn test_degenerate_problem() {
        // Max Z = 3x + 9y
        // s.t. x + 4y <= 8
        //      x + 2y <= 4
        // Solution: x=0, y=2, Z=18.
        // At the point (0,2), slack for 1st constraint is 8-8=0. Slack for 2nd is 4-4=0.
        // This can lead to a basic variable being 0.
        let objective = ObjectiveFunction::new(vec![3.0, 9.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0, 4.0], ConstraintType::LessThanOrEqual, 8.0),
            Constraint::new(vec![1.0, 2.0], ConstraintType::LessThanOrEqual, 4.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert!((solution.objective_value.unwrap_or(0.0) - 18.0).abs() < 1e-6, "Objective was {}", solution.objective_value.unwrap_or(0.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 2);
            // The solution can be x=0, y=2 or x=4, y=0 (Z=12) or x=2, y=1 (Z=15)...
            // Tracing:
            // Tableau: x  y s1 s2 RHS | Basic
            //          1  4  1  0  8  | s1
            //          1  2  0  1  4  | s2
            //         -3 -9  0  0  0  | Z
            // Pivot y (col 1). Ratios: 8/4=2 (s1), 4/2=2 (s2). Tie. Choose s1 (row 0).
            // New Row0 (s1 becomes y): 1/4  1  1/4  0  2
            // Row1 = R1 - 2*NewR0 = [1,2,0,1,4] - [1/2,2,1/2,0,4] = [1/2,0,-1/2,1,0] -> s2 is 0, degenerate.
            // Obj  = Obj - (-9)*NewR0 = [-3,-9,0,0,0] + [9/4,9,9/4,0,18] = [-3/4,0,9/4,0,18]
            // Tableau 2:
            //          1/4  1  1/4  0  2   | y
            //          1/2  0 -1/2  1  0   | s2
            //         -3/4  0  9/4  0  18  | Z
            // Pivot x (col 0). Ratios: (2)/(1/4)=8 (y), (0)/(1/2)=0 (s2). Pivot s2 (row 1).
            // New Row1 (s2 becomes x): 1  0 -1  2  0
            // Row0 = R0 - (1/4)*NewR1 = [1/4,1,1/4,0,2] - [1/4,0,-1/4,1/2,0] = [0,1,1/2,-1/2,2]
            // Obj  = Obj - (-3/4)*NewR1 = [-3/4,0,9/4,0,18] + [3/4,0,-3/4,3/2,0] = [0,0,3/2,3/2,18]
            // Tableau 3 (Optimal):
            //          0  1  1/2 -1/2  2   | y
            //          1  0  -1   2   0   | x
            //          0  0  3/2  3/2 18  | Z
            // Solution: x=0, y=2. Z=18.
            assert!((values[0] - 0.0).abs() < 1e-6, "x value was {}", values[0]); // x = 0
            assert!((values[1] - 2.0).abs() < 1e-6, "y value was {}", values[1]); // y = 2
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }

    #[test]
    fn test_1_var_1_constraint() {
        // Max P = 5x
        // s.t. x <= 10
        // Sol: x=10, P=50
        let objective = ObjectiveFunction::new(vec![5.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0], ConstraintType::LessThanOrEqual, 10.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert_eq!(solution.objective_value, Some(50.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 1);
            assert!((values[0] - 10.0).abs() < 1e-6); // x = 10
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }
    
    #[test]
    fn test_min_1_var_1_constraint() {
        // Min P = 5x
        // s.t. x <= 10 (and x >=0 implied)
        // Sol: x=0, P=0
        let objective = ObjectiveFunction::new(vec![5.0], ObjectiveType::Minimize);
        let constraints = vec![
            Constraint::new(vec![1.0], ConstraintType::LessThanOrEqual, 10.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let mut solver = SimplexSolver::new(&problem).unwrap();
        let solution = solver.solve();

        assert_eq!(solution.status, LPSolutionStatus::Optimal);
        assert_eq!(solution.objective_value, Some(0.0));
        if let Some(values) = solution.variable_values {
            assert_eq!(values.len(), 1);
            assert!((values[0] - 0.0).abs() < 1e-6); // x = 0
        } else {
            panic!("Optimal solution should have variable values.");
        }
    }

    #[test]
    fn test_infeasible_due_to_setup_GreaterThanOrEqual() {
        // Max x s.t. x >= 2. Current solver does not support >=.
        let objective = ObjectiveFunction::new(vec![1.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0], ConstraintType::GreaterThanOrEqual, 2.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let solver_result = SimplexSolver::new(&problem);
        assert!(solver_result.is_err(), "Solver should reject GreaterThanOrEqual constraints.");
    }

    #[test]
    fn test_infeasible_due_to_setup_EqualTo() {
        // Max x s.t. x == 2. Current solver does not support ==.
        let objective = ObjectiveFunction::new(vec![1.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![1.0], ConstraintType::EqualTo, 2.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let solver_result = SimplexSolver::new(&problem);
        assert!(solver_result.is_err(), "Solver should reject EqualTo constraints.");
    }
    
    #[test]
    fn test_infeasible_due_to_setup_negative_rhs() {
        // Max x s.t. -x <= -2 (which is x >= 2)
        // Current solver rejects negative RHS.
        let objective = ObjectiveFunction::new(vec![1.0], ObjectiveType::Maximize);
        let constraints = vec![
            Constraint::new(vec![-1.0], ConstraintType::LessThanOrEqual, -2.0),
        ];
        let problem = LPProblem::new(objective, constraints);
        let solver_result = SimplexSolver::new(&problem);
        assert!(solver_result.is_err(), "Solver should reject negative RHS.");
    }

    // Test for problem that might hit max_iterations if not truly infeasible by structure
    // This example is actually unbounded, but the specific path taken by Simplex might differ.
    // Maximize x1
    // x1 - x2 <= 1
    // -x1 + x2 <= -2 (i.e. x1 - x2 >= 2) -- This is infeasible with the first.
    // Solver should reject due to negative RHS for the second constraint.
    // If we change it to: Max x1, s.t. x1-x2 <= 1, x1-x2 >=2
    // Which is x1-x2 <=1, -x1+x2 <= -2. Rejected.
    //
    // A problem that is accepted but has no solution.
    // Max x1
    // s.t.  x1 <= 1
    //      -x1 <= -2  (i.e. x1 >= 2)
    // This problem would be rejected by `new` due to negative RHS.
    // True infeasibility detection for problems that *pass* `new` is not yet implemented.
    // The current `LPSolutionStatus::Infeasible` is only produced by `max_iterations_reached`.
    // Let's test a problem that will cycle or take many iterations if not handled,
    // leading to `MaxIterationsReached`.
    // (This usually requires specific structures like Klee-Minty, which are complex to set up for basic Simplex)
    // For now, the "infeasible" status primarily means "max_iterations_reached" or "setup error".
    // The existing `test_simplex_unbounded` covers a case where iterations stop due to unboundedness.
    // It's difficult to construct a simple <= problem with non-negative RHS that is infeasible
    // without using techniques not yet implemented (like artificial variables for >= or == constraints
    // that then demonstrate infeasibility).
    // Example: Max x s.t. x <= 1, x >= 2.
    // Transformed for current solver: Max x s.t. x <= 1, -x <= -2.
    // The SimplexSolver::new will return Err for the second constraint's negative RHS.
    // So, this type of infeasibility is caught at setup.

    // A simple problem that is syntactically valid for `new` but might be infeasible:
    // Maximize x1 + x2
    // x1 + x2 <= 1
    // x1 + x2 >= 3  --> This constraint is rejected by `new`.
    // How about: Max x1, s.t. x1 - x2 <= 1, x2 - x1 <= -2 (i.e. x1 - x2 >= 2). Infeasible.
    // The second constraint has negative RHS, so it's rejected by `new`.
    // The current solver is limited in detecting infeasibility beyond setup errors.
    // The `LPSolutionStatus::Infeasible` status is currently only returned if max_iterations is hit.
    // We can't easily construct a problem that is accepted by `new` AND is infeasible AND
    // would be detected as such by the current *basic* Simplex logic without Phase I.
}
