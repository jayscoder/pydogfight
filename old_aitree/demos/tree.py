import old_aitree

def main():
    xml_text = """
    <BehaviorTree>
        <Sequence>
            <Print msg="hello world 0"/>
            <Print msg="hello world 1"/>
            <Print msg="hello world 2"/>
            
        </Sequence>
    </BehaviorTree>
    """
    tree = old_aitree.build_from_xml(xml_text=xml_text)
    print(tree)
    tree.reset()
    tree.tick_once()
    tree.tick_once()
    # for node in tree.tick():
    #     print(node)

if __name__ == '__main__':
    main()
