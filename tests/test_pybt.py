from pybt.node import *
import pybt


class BTGoToLocation(BTAction):
    def update(self) -> Status:
        self.actions.put_nowait((0, 0, 10))
        self.actions.put_nowait((0, 0, 20))
        return Status.SUCCESS


class BTFireMissile(BTAction):
    pass


class BTShouldGoHome(BTCondition):
    pass


class BTGoHome(BTAction):
    pass


def test_pybt_builder():
    builder = pybt.builder.BTBuilder()
    builder.register_bt(BTGoToLocation, BTFireMissile, BTShouldGoHome, BTGoHome)

    xml = """
    <Sequence>
        <BTGoToLocation/>
        <Selector>
            <BTFireMissile/>
            <Sequence>
                <BTShouldGoHome/>
                <BTGoHome/>
            </Sequence>
        </Selector>
    </Sequence>
    """
    node = builder.build_from_xml(xml_data=xml)
    node.tick_once()
    json_data = pybt.utility.bt_to_json(node)
    print(json_data)
    node = builder.build_from_json(json_data=json_data)
    print(pybt.utility.bt_to_xml(node))
    
