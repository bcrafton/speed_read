
'''
m = model(layers=[
conv_block(3,   64, 1, noise=args.noise),
conv_block(64,  64, 2, noise=args.noise),

conv_block(64,  128, 1, noise=args.noise),
conv_block(128, 128, 2, noise=args.noise),

conv_block(128, 256, 1, noise=args.noise),
conv_block(256, 256, 2, noise=args.noise),

avg_pool(4, 4),
dense_block(256, 10, noise=args.noise)
])
'''

l1 = (32 * 32) * (3 * 3 *  3  * 64)
l2 = (32 * 32) * (3 * 3 * 64  * 64)
l3 = (16 * 16) * (3 * 3 * 64  * 128)
l4 = (16 * 16) * (3 * 3 * 128 * 128)
l5 = (8 * 8)   * (3 * 3 * 128 * 256)
l6 = (8 * 8)   * (3 * 3 * 256 * 256)
l7 = (256 * 10) 

print (l1)
print (l2)
print (l3)
print (l4)
print (l5)
print (l6)
print (l7)


