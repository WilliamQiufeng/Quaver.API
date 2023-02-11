using System;
using System.Collections.Generic;

namespace Quaver.API.Maps.Structures
{
    [Serializable]
    public class BookmarkInfo
    {
        public int StartTime { get; set; }

        public string Note { get; set; }

        private sealed class TimeNoteEqualityComparer : IEqualityComparer<BookmarkInfo>
        {
            public bool Equals(BookmarkInfo x, BookmarkInfo y)
            {
                if (ReferenceEquals(x, y)) return true;
                if (ReferenceEquals(x, null)) return false;
                if (ReferenceEquals(y, null)) return false;
                if (x.GetType() != y.GetType()) return false;
                return x.StartTime == y.StartTime && x.Note == y.Note;
            }

            public int GetHashCode(BookmarkInfo obj)
            {
                return HashCode.Combine(obj.StartTime, obj.Note);
            }
        }

        public static IEqualityComparer<BookmarkInfo> ByValueComparer { get; } = new TimeNoteEqualityComparer();
    }
}