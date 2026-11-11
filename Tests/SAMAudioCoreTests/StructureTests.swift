import XCTest
@testable import SAMAudioCore

/// Weight-free structural checks only (the SPM test product's metallib is unreliable; parity lanes live in samaudio-gates).
final class StructureTests: XCTestCase {
    func testSolverMethodsAreNamed() {
        XCTAssertEqual(FlowSolver.Method.midpoint.rawValue, "midpoint")
        XCTAssertEqual(FlowSolver.Method.euler.rawValue, "euler")
    }
}
