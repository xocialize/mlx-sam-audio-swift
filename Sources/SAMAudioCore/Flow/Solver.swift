// Solver.swift — the flow ODE from noise (t = 0) to the separated latent (t = 1). Upstream uses torchdiffeq's fixed-grid
// `midpoint` with step_size 2/32 (16 steps, 32 evaluations of the velocity field).
import Foundation
import MLX

public enum FlowSolver {
    public enum Method: String, Sendable { case midpoint, euler }

    /// `steps` equal steps over [0, 1]. Evaluates after every step so the graph never spans the solve.
    public static func solve(_ model: SAMAudioModel, noise: MLXArray, _ c: Conditioning, steps: Int = 16, method: Method = .midpoint,
                             onStep: ((Int) throws -> Void)? = nil) rethrows -> MLXArray {
        var x = noise
        let dt = 1.0 / Float(steps)
        let b = noise.dim(0)
        for i in 0..<steps {
            let t = Float(i) * dt
            let v0 = model(noisy: x, time: MLXArray([Float](repeating: t, count: b)), c)
            switch method {
            case .euler:
                x = x + dt * v0
            case .midpoint:
                let mid = x + (0.5 * dt) * v0
                x = x + dt * model(noisy: mid, time: MLXArray([Float](repeating: t + 0.5 * dt, count: b)), c)
            }
            eval(x)
            try onStep?(i + 1)
        }
        return x
    }
}
