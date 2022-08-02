import sys
import argparse
import struct
from PIL import Image

def main():
	parser = argparse.ArgumentParser(description='Recolour pixels in BMP images')
	parser.add_argument('--target', metavar='t', type=str, nargs=1, help='A tuple specifying the pixels to recolour')
	parser.add_argument('--recolour', metavar='r', type=str, nargs=1, help="A tuple specifying the new colour for the targeted pixels")
	parser.add_argument('--files', metavar='f', type=str, nargs='+', help="The BMP files to recolour")
	parser.add_argument('-y', '--yes', action='store_true', help="Don't ask if i'm sure")
	
	args = parser.parse_args()

	if (not args.target or not args.recolour or not args.files):
		print("Missing arguments")
		return 1

	target_colour = parse_int_tuple(args.target)
	recolour_colour = parse_int_tuple(args.recolour)

	if (len(target_colour) > 3 or len(recolour_colour) > 3):
		print("Too many elements in tuple")
		return 1

	if (not args.yes):
		print("Change pixels with colour", target_colour, "to", recolour_colour,"in the following files:")
		for file_name in args.files:
			print("    ", file_name)
		inp = input("Would you like to proceed? ")
		if inp != 'y' and inp != 'Y':
			print("Exiting")
			return 0
	i = 0
	for file_name in args.files:
		print(str(round(i / len(args.files) * 100,2)) + "% complete", end='\r', flush=True)
		recolour_pixels(file_name, target_colour, recolour_colour)
		i += 1

	print("Done")
	return 0

def recolour_pixels(file_name, target_colour, recolour_colour):
	img = Image.open(file_name)

	pixels = img.load()

	for i in range(img.size[0]):
		for j in range(img.size[1]):
			if pixels[i, j] == target_colour:
				pixels[i, j] = recolour_colour
	
	img.save(file_name)

def parse_int_tuple(s):
	vals = s[0]
	# Remove spaces
	"".join(vals.split())
	# Split at comma
	vals = vals.split(",")
	# List to tuple
	tup = ()
	for val in vals:
		tup += (int(val),)
	return tup

if __name__ == "__main__":
	main()
