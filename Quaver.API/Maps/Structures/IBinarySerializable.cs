using System.IO;

namespace Quaver.API.Maps.Structures
{
    public interface IBinarySerializable<T> where T : IBinarySerializable<T>
    {
        void Serialize(BinaryWriter writer);
        void Parse(BinaryReader reader);
    }
}