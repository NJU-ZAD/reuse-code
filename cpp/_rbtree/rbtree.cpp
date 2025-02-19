#include "../../hpp/rbtree.hpp"

static void __rbRotateLeft(struct rbNode *node, struct rbRoot *root)
{
	struct rbNode *right = node->rbRight;
	struct rbNode *parent = RB_PARENT(node);

	if ((node->rbRight = right->rbLeft))
		rbSetParent(right->rbLeft, node);
	right->rbLeft = node;

	rbSetParent(right, parent);

	if (parent)
	{
		if (node == parent->rbLeft)
			parent->rbLeft = right;
		else
			parent->rbRight = right;
	}
	else
		root->rbNode = right;
	rbSetParent(node, right);
}

static void __rbRotateRight(struct rbNode *node, struct rbRoot *root)
{
	struct rbNode *left = node->rbLeft;
	struct rbNode *parent = RB_PARENT(node);

	if ((node->rbLeft = left->rbRight))
		rbSetParent(left->rbRight, node);
	left->rbRight = node;

	rbSetParent(left, parent);

	if (parent)
	{
		if (node == parent->rbRight)
			parent->rbRight = left;
		else
			parent->rbLeft = left;
	}
	else
		root->rbNode = left;
	rbSetParent(node, left);
}

void rbInsertColor(struct rbNode *node, struct rbRoot *root)
{
	struct rbNode *parent, *gparent;

	while ((parent = RB_PARENT(node)) && RB_IS_RED(parent))
	{
		gparent = RB_PARENT(parent);

		if (parent == gparent->rbLeft)
		{
			{
				struct rbNode *uncle = gparent->rbRight;
				if (uncle && RB_IS_RED(uncle))
				{
					RB_SET_BLACK(uncle);
					RB_SET_BLACK(parent);
					RB_SET_RED(gparent);
					node = gparent;
					continue;
				}
			}

			if (parent->rbRight == node)
			{
				struct rbNode *tmp;
				__rbRotateLeft(parent, root);
				tmp = parent;
				parent = node;
				node = tmp;
			}

			RB_SET_BLACK(parent);
			RB_SET_RED(gparent);
			__rbRotateRight(gparent, root);
		}
		else
		{
			{
				struct rbNode *uncle = gparent->rbLeft;
				if (uncle && RB_IS_RED(uncle))
				{
					RB_SET_BLACK(uncle);
					RB_SET_BLACK(parent);
					RB_SET_RED(gparent);
					node = gparent;
					continue;
				}
			}

			if (parent->rbLeft == node)
			{
				struct rbNode *tmp;
				__rbRotateRight(parent, root);
				tmp = parent;
				parent = node;
				node = tmp;
			}

			RB_SET_BLACK(parent);
			RB_SET_RED(gparent);
			__rbRotateLeft(gparent, root);
		}
	}

	RB_SET_BLACK(root->rbNode);
}

static void __rbEraseColor(struct rbNode *node, struct rbNode *parent, struct rbRoot *root)
{
	struct rbNode *other;

	while ((!node || RB_IS_BLACK(node)) && node != root->rbNode)
	{
		if (parent->rbLeft == node)
		{
			other = parent->rbRight;
			if (RB_IS_RED(other))
			{
				RB_SET_BLACK(other);
				RB_SET_RED(parent);
				__rbRotateLeft(parent, root);
				other = parent->rbRight;
			}
			if ((!other->rbLeft || RB_IS_BLACK(other->rbLeft)) &&
				(!other->rbRight || RB_IS_BLACK(other->rbRight)))
			{
				RB_SET_RED(other);
				node = parent;
				parent = RB_PARENT(node);
			}
			else
			{
				if (!other->rbRight || RB_IS_BLACK(other->rbRight))
				{
					RB_SET_BLACK(other->rbLeft);
					RB_SET_RED(other);
					__rbRotateRight(other, root);
					other = parent->rbRight;
				}
				rbSetColor(other, RB_COLOR(parent));
				RB_SET_BLACK(parent);
				RB_SET_BLACK(other->rbRight);
				__rbRotateLeft(parent, root);
				node = root->rbNode;
				break;
			}
		}
		else
		{
			other = parent->rbLeft;
			if (RB_IS_RED(other))
			{
				RB_SET_BLACK(other);
				RB_SET_RED(parent);
				__rbRotateRight(parent, root);
				other = parent->rbLeft;
			}
			if ((!other->rbLeft || RB_IS_BLACK(other->rbLeft)) &&
				(!other->rbRight || RB_IS_BLACK(other->rbRight)))
			{
				RB_SET_RED(other);
				node = parent;
				parent = RB_PARENT(node);
			}
			else
			{
				if (!other->rbLeft || RB_IS_BLACK(other->rbLeft))
				{
					RB_SET_BLACK(other->rbRight);
					RB_SET_RED(other);
					__rbRotateLeft(other, root);
					other = parent->rbLeft;
				}
				rbSetColor(other, RB_COLOR(parent));
				RB_SET_BLACK(parent);
				RB_SET_BLACK(other->rbLeft);
				__rbRotateRight(parent, root);
				node = root->rbNode;
				break;
			}
		}
	}
	if (node)
		RB_SET_BLACK(node);
}

void rbErase(struct rbNode *node, struct rbRoot *root)
{
	struct rbNode *child, *parent;
	int color;

	if (!node->rbLeft)
		child = node->rbRight;
	else if (!node->rbRight)
		child = node->rbLeft;
	else
	{
		struct rbNode *old = node, *left;

		node = node->rbRight;
		while ((left = node->rbLeft) != nullptr)
			node = left;

		if (RB_PARENT(old))
		{
			if (RB_PARENT(old)->rbLeft == old)
				RB_PARENT(old)->rbLeft = node;
			else
				RB_PARENT(old)->rbRight = node;
		}
		else
			root->rbNode = node;

		child = node->rbRight;
		parent = RB_PARENT(node);
		color = RB_COLOR(node);

		if (parent == old)
		{
			parent = node;
		}
		else
		{
			if (child)
				rbSetParent(child, parent);
			parent->rbLeft = child;

			node->rbRight = old->rbRight;
			rbSetParent(old->rbRight, node);
		}

		node->rbParentColor = old->rbParentColor;
		node->rbLeft = old->rbLeft;
		rbSetParent(old->rbLeft, node);

		goto color;
	}

	parent = RB_PARENT(node);
	color = RB_COLOR(node);

	if (child)
		rbSetParent(child, parent);
	if (parent)
	{
		if (parent->rbLeft == node)
			parent->rbLeft = child;
		else
			parent->rbRight = child;
	}
	else
		root->rbNode = child;

color:
	if (color == RB_BLACK)
		__rbEraseColor(child, parent, root);
}

static void rbAugmentPath(struct rbNode *node, rbAugmentF func, void *data)
{
	struct rbNode *parent;

up:
	func(node, data);
	parent = RB_PARENT(node);
	if (!parent)
		return;

	if (node == parent->rbLeft && parent->rbRight)
		func(parent->rbRight, data);
	else if (parent->rbLeft)
		func(parent->rbLeft, data);

	node = parent;
	goto up;
}

//*将@node插入树后，更新树以考虑新条目和重新平衡造成的任何损坏
void rbAugmentInsert(struct rbNode *node, rbAugmentF func, void *data)
{
	if (node->rbLeft)
		node = node->rbLeft;
	else if (node->rbRight)
		node = node->rbRight;

	rbAugmentPath(node, func, data);
}

//*在删除节点之前，找到重新平衡路径上最深的节点，该节点在@node被删除后仍然存在
struct rbNode *rbAugmentEraseBegin(struct rbNode *node)
{
	struct rbNode *deepest;

	if (!node->rbRight && !node->rbLeft)
		deepest = RB_PARENT(node);
	else if (!node->rbRight)
		deepest = node->rbLeft;
	else if (!node->rbLeft)
		deepest = node->rbRight;
	else
	{
		deepest = rbNext(node);
		if (deepest->rbRight)
			deepest = deepest->rbRight;
		else if (RB_PARENT(deepest) != node)
			deepest = RB_PARENT(deepest);
	}
	return deepest;
}

//*删除后，更新树以考虑删除的条目和任何重新平衡的损坏
void rbAugmentEraseEnd(struct rbNode *node, rbAugmentF func, void *data)
{
	if (node)
		rbAugmentPath(node, func, data);
}

//*此函数返回树的第一个节点（按排序顺序）
struct rbNode *rbFirst(const struct rbRoot *root)
{
	struct rbNode *n;

	n = root->rbNode;
	if (!n)
		return nullptr;
	while (n->rbLeft)
		n = n->rbLeft;
	return n;
}

struct rbNode *rbLast(const struct rbRoot *root)
{
	struct rbNode *n;

	n = root->rbNode;
	if (!n)
		return nullptr;
	while (n->rbRight)
		n = n->rbRight;
	return n;
}

struct rbNode *rbNext(const struct rbNode *node)
{
	struct rbNode *parent;

	if (RB_PARENT(node) == node)
		return nullptr;

	//*如果有一个右孩子，尽量往下然后往左走
	if (node->rbRight)
	{
		node = node->rbRight;
		while (node->rbLeft)
			node = node->rbLeft;
		return (struct rbNode *)node;
	}

	/*
	 *没有右孩子。
	 *向下和向左的所有节点都比我们小，所以任何“下一个”节点都必须在我们父节点的大体方向上。
	 *沿着树向上；每当祖先是其父节点的右孩子时，继续向上。
	 *当它第一次是其父节点的左孩子时，其父节点是“下一个”节点。
	 */
	while ((parent = RB_PARENT(node)) && node == parent->rbRight)
		node = parent;

	return parent;
}

struct rbNode *rbPrev(const struct rbNode *node)
{
	struct rbNode *parent;

	if (RB_PARENT(node) == node)
		return nullptr;

	//*如果有一个左孩子，尽量往下然后往右走
	if (node->rbLeft)
	{
		node = node->rbLeft;
		while (node->rbRight)
			node = node->rbRight;
		return (struct rbNode *)node;
	}

	//*没有左孩子，一直向上，直到找到一个祖先，它是其父节点的右孩子
	while ((parent = RB_PARENT(node)) && node == parent->rbLeft)
		node = parent;

	return parent;
}

void rbReplaceNode(struct rbNode *victim, struct rbNode *nw, struct rbRoot *root)
{
	struct rbNode *parent = RB_PARENT(victim);

	//*设置周围节点指向替换位置
	if (parent)
	{
		if (victim == parent->rbLeft)
			parent->rbLeft = nw;
		else
			parent->rbRight = nw;
	}
	else
	{
		root->rbNode = nw;
	}
	if (victim->rbLeft)
		rbSetParent(victim->rbLeft, nw);
	if (victim->rbRight)
		rbSetParent(victim->rbRight, nw);

	//*将指针/颜色从victim复制到替换位置
	*nw = *victim;
}