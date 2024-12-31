class TreeNode:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# 运算符优先级定义
def get_precedence(op):
    if op in ('+', '-'):
        return 1
    elif op in ('*', '/'):
        return 2
    return 0

# 构建表达式树的函数
def construct_expression_tree(infix):
    def apply_operator(operators, operands):
        operator = operators.pop()
        right = operands.pop()
        left = operands.pop()
        node = TreeNode(operator)
        node.left = left
        node.right = right
        operands.append(node)

    operators = []  # 存放运算符的栈
    operands = []   # 存放操作数（节点）的栈

    i = 0
    while i < len(infix):
        char = infix[i]

        # 跳过空格
        if char == ' ':
            i += 1
            continue

        # 如果是数字，处理多位数
        if char.isdigit():
            num = []
            while i < len(infix) and infix[i].isdigit():
                num.append(infix[i])
                i += 1
            operands.append(TreeNode(''.join(num)))
            continue

        # 如果是左括号，直接压入运算符栈
        if char == '(':
            operators.append(char)

        # 如果是右括号，处理括号内的所有内容
        elif char == ')':
            while operators and operators[-1] != '(':
                apply_operator(operators, operands)
            operators.pop()  # 弹出左括号

        # 如果是运算符，处理优先级
        elif char in ('+', '-', '*', '/'):
            while (operators and operators[-1] != '(' and
                   get_precedence(operators[-1]) >= get_precedence(char)):
                apply_operator(operators, operands)
            operators.append(char)

        i += 1

    # 处理剩余的运算符
    while operators:
        apply_operator(operators, operands)

    return operands[-1]

# 中序遍历打印表达式树
def inorder_traversal(root):
    if root is not None:
        if root.left or root.right:
            print('(', end='')
        inorder_traversal(root.left)
        print(root.value, end=' ')
        inorder_traversal(root.right)
        if root.left or root.right:
            print(')', end='')

# 示例用法
infix = "3 + 4 * 2 / ( 1 - 5 )"
tree = construct_expression_tree(infix)
print("中缀表达式的树结构:")
inorder_traversal(tree)
