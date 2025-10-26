"""
Basic usage example for Cumulus SDK
"""

from cumulus.sdk import CumulusClient, remote, gpu


def main():
    # Create client
    client = CumulusClient(server_url="http://your-server:8080")
    
    # Example 1: Simple function execution
    def simple_calculation():
        import math
        return {
            "pi": math.pi,
            "sqrt_2": math.sqrt(2),
            "factorial_10": math.factorial(10)
        }
    
    print("🚀 Running simple calculation...")
    result = client.run(
        func=simple_calculation,
        gpu_memory=0.1,  # 10% of GPU memory
        duration=300,    # 5 minutes
        requirements=["math"]  # No external requirements needed
    )
    print(f"Result: {result}")
    
    # Example 2: Using decorators
    @remote(client, gpu_memory=0.5, duration=1800)
    def gpu_intensive_task():
        import numpy as np
        # Simulate GPU-intensive computation
        data = np.random.rand(1000, 1000)
        result = np.linalg.inv(data)
        return {"shape": result.shape, "mean": float(np.mean(result))}
    
    print("\n🚀 Running GPU-intensive task...")
    result = gpu_intensive_task()
    print(f"Result: {result}")
    
    # Example 3: Using @gpu decorator
    @gpu(client, memory=0.8, duration=3600)
    def machine_learning_task():
        import numpy as np
        from sklearn.linear_model import LinearRegression
        
        # Generate sample data
        X = np.random.rand(100, 1)
        y = 2 * X.flatten() + 1 + np.random.rand(100) * 0.1
        
        # Train model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        return {
            "coefficient": float(model.coef_[0]),
            "intercept": float(model.intercept_),
            "mse": float(np.mean((y - predictions) ** 2))
        }
    
    print("\n🚀 Running machine learning task...")
    result = machine_learning_task()
    print(f"Result: {result}")


if __name__ == "__main__":
    main()
