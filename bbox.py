# 2D point container
class p2d: 
  def __init__(self,x=0,y=0):
    self.x = x
    self.y = y
  def __str__ (self):
    return '(' + str(self.x) + ', ' + str(self.y) + ')'

#
class bbox:
  """ bounding box class
  """
  def __init__ (self,left=0,right=0,bottom=0,top=0):
    self.left,self.right,self.bottom,self.top=left,right,bottom,top
  def __str__ (self):
    return str(self.left) + ' ' + str(self.right) + ' ' + str(self.bottom) + ' ' + str(self.top)
  def area (self):
    # get the area
    return (self.right - self.left) * (self.top-self.bottom)
  def width (self):
    return self.right - self.left + 1
  def height (self):
    return self.top - self.bottom + 1
  def intersect (self, box2):
    # check if this box intersects with box2
    return box2.right >= self.left and box2.left <= self.right and box2.top >= self.bottom and box2.bottom <= self.top
  def getunion (self, box2):
    # get bbox of union
    bnew = bbox()
    bnew.left=min(self.left,box2.left)
    bnew.right=max(self.right,box2.right)
    bnew.bottom=min(self.bottom,box2.bottom)
    bnew.top=max(self.top,box2.top)
    return bnew
  def getintersection (self,box2):
    # get bbox of intersection
    bnew = bbox()
    if not self.intersect(box2): return bnew
    bnew.left = max(self.left,box2.left)
    bnew.right = min(self.right,box2.right)
    bnew.bottom = max(self.bottom,box2.bottom)
    bnew.top = min(self.top,box2.top)
    return bnew
