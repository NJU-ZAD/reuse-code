#ifndef _LINUX_RBTREE_H
#define _LINUX_RBTREE_H

#if defined(CONTAINER_OF)
#undef CONTAINER_OF
#define CONTAINER_OF(ptr, type, member) (                    \
	{                                                        \
		const decltype(((type *)0)->member) *__mptr = (ptr); \
		(type *)((char *)__mptr - OFFSETOF(type, member));   \
	})
#else
#define CONTAINER_OF(ptr, type, member) (                    \
	{                                                        \
		const decltype(((type *)0)->member) *__mptr = (ptr); \
		(type *)((char *)__mptr - OFFSETOF(type, member));   \
	})
#endif

#if defined(OFFSETOF)
#undef OFFSETOF
#define OFFSETOF(TYPE, MEMBER) ((size_t) & ((TYPE *)0)->MEMBER)
#else
#define OFFSETOF(TYPE, MEMBER) ((size_t) & ((TYPE *)0)->MEMBER)
#endif

struct rbNode
{
	unsigned long rbParentColor;
#define RB_RED 0
#define RB_BLACK 1
	struct rbNode *rbRight;
	struct rbNode *rbLeft;
};

struct rbRoot
{
	struct rbNode *rbNode;
};

#define RB_PARENT(r) ((struct rbNode *)((r)->rbParentColor & ~3))
#define RB_COLOR(r) ((r)->rbParentColor & 1)
#define RB_IS_RED(r) (!RB_COLOR(r))
#define RB_IS_BLACK(r) RB_COLOR(r)
#define RB_SET_RED(r)             \
	do                            \
	{                             \
		(r)->rbParentColor &= ~1; \
	} while (0)
#define RB_SET_BLACK(r)          \
	do                           \
	{                            \
		(r)->rbParentColor |= 1; \
	} while (0)

static inline void rbSetParent(struct rbNode *rb, struct rbNode *p)
{
	rb->rbParentColor = (rb->rbParentColor & 3) | (unsigned long)p;
}
static inline void rbSetColor(struct rbNode *rb, int color)
{
	rb->rbParentColor = (rb->rbParentColor & ~1) | color;
}

#define RB_ENTRY(ptr, type, member) CONTAINER_OF(ptr, type, member)

#define RB_EMPTY_ROOT(root) ((root)->rbNode == nullptr)
#define RB_EMPTY_NODE(node) (RB_PARENT(node) == node)
#define RB_CLEAR_NODE(node) (rbSetParent(node, node))

static inline void rbInitNode(struct rbNode *rb)
{
	rb->rbParentColor = 0;
	rb->rbRight = nullptr;
	rb->rbLeft = nullptr;
	RB_CLEAR_NODE(rb);
}

extern void rbInsertColor(struct rbNode *, struct rbRoot *);
extern void rbErase(struct rbNode *, struct rbRoot *);

typedef void (*rbAugmentF)(struct rbNode *node, void *data);

extern void rbAugmentInsert(struct rbNode *node, rbAugmentF func, void *data);
extern struct rbNode *rbAugmentEraseBegin(struct rbNode *node);
extern void rbAugmentEraseEnd(struct rbNode *node, rbAugmentF func, void *data);

//*在树中查找逻辑上的前继节点和后继节点
extern struct rbNode *rbNext(const struct rbNode *);
extern struct rbNode *rbPrev(const struct rbNode *);
extern struct rbNode *rbFirst(const struct rbRoot *);
extern struct rbNode *rbLast(const struct rbRoot *);

//*快速替换单个节点，无需删除/重新平衡/添加/重新平衡
extern void rbReplaceNode(struct rbNode *victim, struct rbNode *nw, struct rbRoot *root);

static inline void rbLinkNode(struct rbNode *node, struct rbNode *parent, struct rbNode **rbLink)
{
	node->rbParentColor = (unsigned long)parent;
	node->rbLeft = node->rbRight = nullptr;
	*rbLink = node;
}

#endif