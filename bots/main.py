from stock_bot import StockBot

def main():
    attribute_sets = [("AAPL", "Apple Inc"), ("AMD", "Advanced Micro Devices, Inc")]

    instances = (StockBot(attr1, attr2) for attr1, attr2 in attribute_sets)
    results = [instance.launch_bot() for instance in instances]
    for i, result in enumerate(results):
        print(f"Result from instance {i + 1}: {result}")

if __name__ == "__main__":
    main()
