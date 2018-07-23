import unittest

import rockstar as rs

class RockstarTest(unittest.TestCase):
    def test_tokenize(self):
        t = rs.tokenize
        self.assertEqual(
            t(' hello   world '),
            ['hello', 'world'])
        self.assertEqual(
            t('tau is 6.28'),
            ['tau', 'is', '6.28'])
        self.assertEqual(
            t("Tommy ain't nobody"),
            ['Tommy', "ain't", 'nobody'])
        self.assertEqual(
            t('Alice stated "Bob said \'yes\'" and laughed'),
            ['Alice', 'stated', '"Bob said \'yes\'"', 'and', 'laughed'])
        self.assertEqual(
            t('My cat is round and fluffy'),
            ['My', 'cat', 'is', 'round and fluffy'])
        self.assertEqual(
            t('James Earl Jones was Darth Vader'),
            ['James', 'Earl', 'Jones', 'was', 'Darth Vader'])
        self.assertEqual(
            t('My cat says she\'s hungry'),
            ['My', 'cat', 'says', "she's hungry"])

    def test_parse_literal(self):
        p = lambda tokens: rs.parse_expression(tokens, rs.ParseContext())
        self.assertEqual(p(['77']), rs.ConstantExpression(77))
        self.assertEqual(p(['-1.25']), rs.ConstantExpression(-1.25))
        self.assertEqual(p(['nowhere']), rs.ConstantExpression(rs.NULL))
        self.assertEqual(p(['"hi there"']), rs.ConstantExpression('hi there'))

    def test_assignment(self):
        p = lambda tokens: rs.parse_expression(tokens, rs.ParseContext())
        self.assertEqual(
            p(['Put', '"all your money"', 'into', 'Planet', 'Express']),
            rs.AssignmentExpression('Planet Express', rs.ConstantExpression('all your money')))
        self.assertEqual(
            p(['Put', 'your', 'bitcoins', 'into', 'my', 'account']),
            rs.AssignmentExpression('my account', rs.VariableExpression('your bitcoins')))
        self.assertEqual(
            p(['Put', 'nothing', 'into', 'your', 'hopes']),
            rs.AssignmentExpression('your hopes', rs.VariableExpression(rs.NULL)))

    def test_parse_poetic_assignment(self):
        p = lambda tokens: rs.parse_expression(tokens, rs.ParseContext())
        self.assertEqual(
            p(['X', 'is', 'true']),
            rs.AssignmentExpression('X', rs.ConstantExpression(True)))
        self.assertEqual(
            p(['THE', 'world', 'is', 'nothing']),
            rs.AssignmentExpression('the world', rs.ConstantExpression(rs.NULL)))
        self.assertEqual(
            p(['My', 'life', 'is', 'about to change']),
            rs.AssignmentExpression('my life', rs.ConstantExpression(526)))
        self.assertEqual(
            p(['Jane', 'Doe', 'was', '30']),
            rs.AssignmentExpression('Jane Doe', rs.ConstantExpression(30)))
        self.assertEqual(
            p(['My', 'dreams', 'were',
               "ice. A life unfulfilled; wakin' everybody up, taking booze. And pills--"]),
            rs.AssignmentExpression('my dreams', rs.ConstantExpression(3.1415926535)))


if __name__ == '__main__':
    unittest.main()
