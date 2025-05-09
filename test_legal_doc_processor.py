from legal_doc_processor import create_legal_doc_processor, GraphState, Node, process_document_chunk

def test_legal_doc_processing():
    # Create the processor
    processor = create_legal_doc_processor()
    
    # Initialize state with base tax regulation node
    initial_state = GraphState(
        nodes={
            "node_1": Node(
                content="Corporate tax deductions are allowable for ordinary and necessary business expenses incurred during the taxable year.",
                node_type="tax_regulation"
            )
        }
    )
    
    # Test chunks representing different aspects of tax regulations
    chunks = [
        # Chunk 1: Basic definition and criteria (complementary information)
        """
        Business expenses must be both ordinary and necessary to qualify for tax deductions. 
        An ordinary expense is one that is common and accepted in the industry. 
        A necessary expense is one that is helpful and appropriate for the business. 
        The expense must be directly related to the business and not be lavish or extravagant under the circumstances.
        """,
        
        # Chunk 2: Capital expenditure information (new category with relationship)
        """
        Capital expenditures cannot be deducted as business expenses. These are costs that are part of your investment 
        in your business and are generally capitalized. This includes business startup costs, business assets, and 
        improvements that increase the value of your property, make it more useful, or lengthen its life.
        """,
        
        # Chunk 3: Contradiction with conditions (startup costs exception)
        """
        Despite earlier regulations, recent tax court rulings have determined that certain business startup costs 
        can be deducted as ordinary business expenses in the first year of operation, rather than being capitalized 
        over multiple years. This applies specifically to startup costs under $5,000.
        """,
        
        # Chunk 4: Large chunk causing overflow (detailed deduction types)
        """
        Deductible business expenses include: rent for property used in business; salaries and wages for employees; 
        repairs and maintenance; utilities; office supplies; business travel; business meals (limited to 50% deductibility); 
        business insurance; legal and professional fees; retirement plans for employees; interest on business loans; 
        depreciation on business equipment; advertising costs; business taxes and licenses; employee benefits programs; 
        insurance for employees. Each of these categories has specific requirements and limitations that must be met 
        for the expense to qualify as a deduction. The requirements vary by expense type and may be subject to annual 
        changes in tax law. Documentation requirements also vary by expense category.
        """,
        
        # Chunk 5: Conditional contradiction (meal expenses during COVID)
        """
        Business meal expenses are 100% deductible when purchased from restaurants during tax years 2021 and 2022, 
        as a temporary measure to help the restaurant industry recover from the COVID-19 pandemic. For all other 
        tax years, the 50% limitation on meal deductions applies as per standard regulations.
        """,
        
        # Chunk 6: Vehicle expenses (detailed subcategory)
        """
        Business vehicle expenses can be deducted using two methods: the standard mileage rate method or the actual 
        expense method. The standard mileage rate is set annually by the IRS and for 2023 is 65.5 cents per mile. 
        The actual expense method allows deduction of all vehicle operating expenses based on the percentage of 
        business use, including gas, insurance, repairs, and depreciation.
        """,
        
        # Chunk 7: Information removal
        """
        Effective January 1, 2025, the section regarding depreciation of business equipment should be disregarded 
        as it has been superseded by new regulations that will be published separately.
        """,
        
        # Chunk 8: Multiple information types (entertainment and QIP)
        """
        Entertainment expenses are no longer deductible under any circumstances after the Tax Cuts and Jobs Act of 2017. 
        Qualified improvement property (QIP) is now eligible for bonus depreciation as a technical correction to the 
        Tax Cuts and Jobs Act. QIP includes improvements to the interior of nonresidential property but excludes 
        enlargements, elevators/escalators, and internal structural framework.
        """,
        
        # Chunk 9: Cross-reference information
        """
        Section 179 deduction allows businesses to deduct the full purchase price of qualifying equipment and software 
        purchased during the tax year, subject to limitations. This provision relates directly to the capitalization 
        rules mentioned in the capital expenditures section but creates an exception to those general rules.
        """,
        
        # Chunk 10: Conditional relationship
        """
        A de minimis safe harbor election can be made annually to expense certain items costing less than a specified 
        threshold: $2,500 per item if the business doesn't have an applicable financial statement (AFS), or $5,000 
        per item with an AFS. This applies only to taxpayers with qualifying accounting procedures in place at the 
        beginning of the tax year.
        """
    ]
    
    # Process each chunk and track the evolution of the knowledge graph
    current_state = initial_state
    for i, chunk in enumerate(chunks, 1):
        print(f"\nProcessing Chunk {i}...")
        current_state = process_document_chunk(processor, current_state, chunk)
        
        # Print the current state of the knowledge graph
        print(f"\nKnowledge Graph after Chunk {i}:")
        if current_state.nodes:
            for node_id, node in current_state.nodes.items():
                print(f"\nNode {node_id} ({node.node_type}):")
                print(f"Content: {node.content}")
                if node.relationships:
                    print("Relationships:", node.relationships)
                if node.metadata:
                    print("Metadata:", node.metadata)
        else:
            print("No nodes in the knowledge graph.")
        print("\n" + "="*80)

if __name__ == "__main__":
    test_legal_doc_processing() 